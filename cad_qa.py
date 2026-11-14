"""Answer quantity questions about DXF floor plans with entity-level evidence.

Works on layered architectural DXFs such as the Swiss Dwellings floors in the
challenge data (layers WALL, DOOR, WINDOW, ROOM_OUTLINE, ROOM_TEXT, FIXTURE,
UNIT_OUTLINE). Every answer cites the DXF entity handles it was computed from,
with bounding boxes in drawing units (metres), so a human can check it.

Rules are generic drawing conventions, not per-file lookups:
- a door is one closed outline on the DOOR layer (a double door is one outline
  with two swing arcs); a window is one closed outline on the WINDOW layer;
- a room is a closed ROOM_OUTLINE polygon; its name and number are the
  ROOM_TEXT labels inside it; its net area is the polygon area;
- an apartment is a UNIT_OUTLINE polygon labelled e.g. "Whg. A" / "App. A";
- fixtures: a long curved basin is a bathtub, a curved bowl with a drain
  circle is a washbasin/sink, a curved bowl without a drain is a toilet, and a
  straight-sided tray with a drain is a shower.

Questions are understood in English, German, French and Italian. Unsupported
questions return status "not_found" instead of a guess.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

OUTDOOR_WORDS = ("balkon", "balcon", "balcone", "loggia", "terrasse", "terrazza", "terrace",
                 "sitzplatz", "veranda", "patio", "garten", "jardin", "giardino")
SHAFT_VOID_WORDS = ("schacht", "luftraum", "gaine", "vide", "cavedio", "vuoto", "shaft", "void")
UNIT_LABEL_RE = re.compile(r"^(?:whg|app|apt|wohnung|appartement|appartamento|unit|logement)\.?\s*([A-Z0-9]+)$", re.I)
ROOM_NUMBER_RE = re.compile(r"^[A-Z]?\d+(?:\.\d+)*$")
AREA_LABEL_RE = re.compile(r"^\d+(?:[.,]\d+)?\s*m(?:²|2)$")


# --------------------------------------------------------------------------- geometry

def _points(entity: Any) -> List[Point]:
    kind = entity.dxftype()
    if kind == "LWPOLYLINE":
        return [(float(x), float(y)) for x, y, *_ in entity.get_points()]
    if kind == "POLYLINE":
        return [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in entity.vertices]
    if kind in {"CIRCLE", "ARC"}:
        c, r = entity.dxf.center, float(entity.dxf.radius)
        a0, a1 = (0.0, 360.0) if kind == "CIRCLE" else (float(entity.dxf.start_angle), float(entity.dxf.end_angle))
        if a1 < a0:
            a1 += 360.0
        return [(c.x + r * math.cos(math.radians(a0 + (a1 - a0) * i / 16)),
                 c.y + r * math.sin(math.radians(a0 + (a1 - a0) * i / 16))) for i in range(17)]
    if kind in {"TEXT", "MTEXT"}:
        p = entity.dxf.insert
        return [(float(p.x), float(p.y))]
    return []


def _is_closed(entity: Any, pts: Sequence[Point]) -> bool:
    if getattr(entity, "closed", False) or getattr(entity, "is_closed", False):
        return True
    return len(pts) >= 4 and math.dist(pts[0], pts[-1]) < 1e-6


def _ring(pts: Sequence[Point]) -> List[Point]:
    pts = list(pts)
    if len(pts) > 1 and math.dist(pts[0], pts[-1]) < 1e-9:
        pts = pts[:-1]
    return pts


def _area(pts: Sequence[Point]) -> float:
    ring = _ring(pts)
    return abs(sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
                   for i in range(len(ring)))) / 2.0


def _inside(point: Point, pts: Sequence[Point]) -> bool:
    ring = _ring(pts)
    x, y = point
    hit = False
    for i in range(len(ring)):
        (x1, y1), (x2, y2) = ring[i], ring[(i + 1) % len(ring)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def _bbox(pts: Iterable[Point]) -> List[float]:
    pts = list(pts)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]


def _contains(outer: Sequence[float], inner: Sequence[float], tol: float = 0.01) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _obb(pts: Sequence[Point]) -> Tuple[float, float]:
    """(short side, long side) of the minimum-area bounding rectangle."""
    ring = sorted(set((round(x, 6), round(y, 6)) for x, y in pts))
    if len(ring) < 3:
        return (0.0, math.dist(ring[0], ring[-1]) if len(ring) == 2 else 0.0)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in ring:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(ring):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    best = None
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(-ang), math.sin(-ang)
        xs = [x * c - y * s for x, y in hull]
        ys = [x * s + y * c for x, y in hull]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if best is None or w * h < best[0]:
            best = (w * h, min(w, h), max(w, h))
    return best[1], best[2]


# --------------------------------------------------------------------------- index

def _norm(text: Any) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).casefold()
    return "".join(ch for ch in text if not unicodedata.combining(ch)).strip()


def _record(entity: Any, pts: Sequence[Point], **extra: Any) -> Dict[str, Any]:
    return {"handle": str(entity.dxf.handle), "layer": str(entity.dxf.layer),
            "type": entity.dxftype(), "bbox": _bbox(pts), **extra}


def build_floor_index(file_path: str | Path) -> Dict[str, Any]:
    """Read a layered DXF floor plan into rooms, units, openings and fixtures."""
    try:
        import ezdxf
    except ImportError as exc:
        raise RuntimeError("DXF parsing requires ezdxf. Install it with: pip install ezdxf") from exc
    path = Path(file_path).resolve()
    if path.suffix.casefold() == ".dwg":
        try:
            from ezdxf.addons import odafc
            drawing = odafc.readfile(path)
        except Exception as exc:
            raise RuntimeError(
                "DWG question answering requires the ODA File Converter and ezdxf's odafc add-on. "
                f"Conversion failed: {exc}"
            ) from exc
    else:
        drawing = ezdxf.readfile(path)
    msp = drawing.modelspace()

    by_layer: Dict[str, List[Tuple[Any, List[Point]]]] = {}
    for entity in msp:
        pts = _points(entity)
        if pts:
            by_layer.setdefault(str(entity.dxf.layer).upper(), []).append((entity, pts))

    doors = [_record(e, p) for e, p in by_layer.get("DOOR", []) if e.dxftype() != "ARC" and _is_closed(e, p)]
    windows = [_record(e, p) for e, p in by_layer.get("WINDOW", []) if _is_closed(e, p)]

    # Rooms and their labels.
    texts = [(e, p[0], str(e.dxf.text if e.dxftype() == "TEXT" else e.plain_text()).strip())
             for e, p in by_layer.get("ROOM_TEXT", [])]
    outlines = [(e, p, _area(p)) for e, p in by_layer.get("ROOM_OUTLINE", []) if len(_ring(p)) >= 3]
    # Each label belongs to the smallest room outline that contains it (outlines can nest,
    # e.g. a lift shaft drawn inside a stair hall).
    owner: Dict[int, List[Tuple[Any, Point, str]]] = {}
    for t, pt, txt in texts:
        hits = [(a, i) for i, (e, p, a) in enumerate(outlines) if _inside(pt, p)]
        if hits:
            owner.setdefault(min(hits)[1], []).append((t, pt, txt))
    rooms = []
    for i, (e, p, _) in enumerate(outlines):
        own = owner.get(i, [])
        labels = [(t, txt) for t, pt, txt in own]
        numbers = [txt for _, txt in labels if ROOM_NUMBER_RE.match(txt)]
        names = [txt for _, txt in labels if not ROOM_NUMBER_RE.match(txt)
                 and not AREA_LABEL_RE.match(txt) and not UNIT_LABEL_RE.match(txt)]
        anchor = next((pt for t, pt, txt in own if ROOM_NUMBER_RE.match(txt)), None)
        rooms.append(_record(e, p, polygon=_ring(p), area=round(_area(p), 4), anchor=anchor,
                             number=numbers[0] if numbers else None,
                             name=names[0] if names else None,
                             label_handles=[str(t.dxf.handle) for t, _ in labels]))

    units = []
    for e, p in by_layer.get("UNIT_OUTLINE", []):
        label_hits = [(t, UNIT_LABEL_RE.match(txt)) for t, pt, txt in texts if _inside(pt, p)]
        label = next((m.group(1).upper() for _, m in label_hits if m), None)
        units.append(_record(e, p, polygon=_ring(p), label=label))
    for room in rooms:
        point = room.pop("anchor") or (sum(x for x, _ in room["polygon"]) / len(room["polygon"]),
                                       sum(y for _, y in room["polygon"]) / len(room["polygon"]))
        room["unit"] = next((u["label"] for u in units if u["label"] and _inside(point, u["polygon"])), None)

    # Fixtures.
    fixture_items = [(e, p, _bbox(p)) for e, p in by_layer.get("FIXTURE", [])]
    drains = [(e, b) for e, p, b in fixture_items if e.dxftype() == "CIRCLE" and float(e.dxf.radius) <= 0.06]
    fixtures = []
    for e, p, b in fixture_items:
        if e.dxftype() == "CIRCLE" or not _is_closed(e, p):
            continue
        has_drain = any(_contains(b, db, 0.0) for _, db in drains)
        curved = len(_ring(p)) >= 24
        short, long = _obb(p)
        # A rounded rectangle fills ~95% of its bounding box, an ellipse ~79%.
        rounded_rect = curved and short * long > 0 and _area(p) / (short * long) > 0.88
        if rounded_rect and long >= 1.2:
            kind = "bathtub"
        elif curved:
            kind = "washbasin" if has_drain else "toilet"
        elif has_drain and len(_ring(p)) == 4 and not any(
                _contains(b, ib, 0.0) for ie, ip, ib in fixture_items
                if ie is not e and ie.dxftype() == "LWPOLYLINE" and len(_ring(ip)) >= 24):
            kind = "shower"
        else:
            continue
        fixtures.append(_record(e, p, kind=kind))
    showers = [f for f in fixtures if f["kind"] == "shower"]
    nested = {id(f) for f in showers for g in showers if f is not g and _contains(g["bbox"], f["bbox"], 0.0)
              and (f["bbox"] != g["bbox"] or f["handle"] > g["handle"])}
    fixtures = [f for f in fixtures if id(f) not in nested]

    return {"source": {"path": str(path), "format": path.suffix.lstrip(".").upper(), "units": "m"},
            "rooms": rooms, "units": units, "doors": doors, "windows": windows, "fixtures": fixtures}


# --------------------------------------------------------------------------- questions

COUNT_TARGETS = [  # order matters: first match wins
    ("apartments", ("apartment", "residential unit", "wohnung", "appartement", "appartament", "logement")),
    ("bathtubs", ("bathtub", "badewanne", "baignoire", "vasche da bagno", "vasca")),
    ("showers", ("shower", "dusche", "douche", "docc")),
    ("washbasins", ("washbasin", "lavabo", "lavabi", "waschbecken", "lavandin")),
    ("toilets", ("toilet", "wc")),
    ("windows", ("window", "fenster", "fenetre", "finestr")),
    ("doors", ("door", "tur", "porte", "porta")),
]
ROOM_WORDS = ("room", "raum", "raume", "piece", "locale", "locali", "zimmer")
AREA_WORDS = ("area", "flache", "surface", "superficie")
NUMBERED_WORDS = ("numbered", "nummeriert", "numerot", "numerat")
APARTMENT_RE = re.compile(r"\b(?:apartment|wohnung|appartement|appartamento|logement|unit)\s+([A-Z0-9])\b", re.I)
QUOTES_OPEN, QUOTES_CLOSE = "'\"‘“„«‚", "'\"’”“»‘"


def _quoted(question: str, value: str) -> bool:
    pattern = rf"[{re.escape(QUOTES_OPEN)}]\s*{re.escape(value)}\s*[{re.escape(QUOTES_CLOSE)}]"
    return re.search(pattern, question, re.I) is not None


def _evidence(index: Dict[str, Any], records: Sequence[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    return {"source": index["source"]["path"], **extra,
            "elements": [{k: r[k] for k in ("handle", "layer", "bbox") if k in r}
                         | {k: r[k] for k in ("kind", "name", "number", "area", "unit") if r.get(k) is not None}
                         for r in records]}


def _answer(value: Any, unit: str, method: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(value, float):
        value = round(value, 2)
    return {"status": "answered", "value": value, "unit": unit, "method": method, "evidence": [evidence]}


def _not_found(reason: str) -> Dict[str, Any]:
    return {"status": "not_found", "value": None, "reason": reason, "evidence": []}


def _is_excluded(room: Dict[str, Any]) -> bool:
    name = _norm(room.get("name"))
    return any(word in name for word in OUTDOOR_WORDS + SHAFT_VOID_WORDS)


def answer_cad_question(question: str, index: Dict[str, Any]) -> Dict[str, Any]:
    """Answer one count/area question from ``build_floor_index`` output."""
    q = _norm(question)
    rooms = index["rooms"]
    names = sorted({r["name"] for r in rooms if r.get("name")}, key=len, reverse=True)
    label = next((n for n in names if _quoted(question, n)), None)
    numbers = sorted({r["number"] for r in rooms if r.get("number")}, key=len, reverse=True)
    number = next((n for n in numbers if re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", question)), None)
    apartment = APARTMENT_RE.search(question)
    asks_area = any(word in q for word in AREA_WORDS) or "m²" in question or " m2" in q

    if asks_area:
        if apartment:
            letter = apartment.group(1).upper()
            members = [r for r in rooms if r.get("unit") == letter and r.get("number") and not _is_excluded(r)]
            unit = next((u for u in index["units"] if u["label"] == letter), None)
            if not members or unit is None:
                return _not_found(f"No apartment labelled {letter} with numbered rooms.")
            return _answer(sum(r["area"] for r in members), "m2", "sum of room polygon areas in apartment, outdoor excluded",
                           _evidence(index, members, apartment=letter, apartment_outline=unit["handle"]))
        if number:
            room = next(r for r in rooms if r.get("number") == number)
            return _answer(room["area"], "m2", "room polygon area", _evidence(index, [room]))
        if label:
            members = [r for r in rooms if r.get("name") == label]
            return _answer(sum(r["area"] for r in members), "m2", f"sum of areas of rooms named {label!r}",
                           _evidence(index, members))
        if any(word in q for word in NUMBERED_WORDS):
            members = [r for r in rooms if r.get("number") and not _is_excluded(r)]
            return _answer(sum(r["area"] for r in members), "m2", "sum of numbered indoor room areas",
                           _evidence(index, members))
        return _not_found("Area question did not name a room, label, apartment or floor total.")

    if label and any(re.search(rf"\b{w}", q) for w in ROOM_WORDS):
        members = [r for r in rooms if r.get("name") == label]
        return _answer(len(members), "count", f"rooms whose label is {label!r}", _evidence(index, members))

    for target, words in COUNT_TARGETS:
        if any(re.search(rf"\b{re.escape(w)}", q) for w in words):
            if target == "apartments":
                items = [u for u in index["units"] if u["label"]]
            elif target in {"doors", "windows"}:
                items = index[target]
            else:
                kind = target[:-1] if target != "washbasins" else "washbasin"
                items = [f for f in index["fixtures"] if f["kind"] == kind]
            return _answer(len(items), "count", f"{target} drawn on the plan", _evidence(index, items))

    return _not_found("Question type not supported by the CAD answerer yet.")


def find_cad_twin(question_path: str | Path, data_dir: str | Path) -> Optional[Path]:
    """Return the DXF drawing for a question's file (same stem, any extension)."""
    path = Path(question_path)
    rel = path.relative_to("data") if path.parts and path.parts[0] == "data" else path
    candidate = (Path(data_dir) / rel).with_suffix(".dxf")
    return candidate if candidate.exists() else None
