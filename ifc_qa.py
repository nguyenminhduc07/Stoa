"""Answer element counts and wall/slab quantities from an IFC model, with GUIDs as evidence.

Handles, for the whole model or one storey named in the question:
- counts of doors, windows, rooms (spaces), walls, columns, beams, stairs,
  slabs and sanitary fixtures;
- total wall length, total wall net side area, total wall net volume and
  total slab net volume.

Rules (generic IFC semantics, no per-file lookups):
- an element belongs to the storey that contains it, walking up the spatial
  structure / aggregation (so stair flights inside a stair count with it);
- a stair is an IfcStair that is not itself part of another IfcStair
  (some exporters split one stair into many IfcStair parts);
- sanitary fixtures are IfcSanitaryTerminal (IFC4) or IfcFlowTerminal typed
  by IfcSanitaryTerminalType (IFC2x3);
- quantities come from the model's authored base quantities (Qto_*), scaled
  to metres / m2 / m3 with the model's project units. If a wall or slab has no
  authored quantity, its volume is computed from geometry; length and side
  area are then reported as not found rather than guessed.

Questions are understood in English, Danish, Estonian, German, French and Finnish.
Room net floor area questions stay with bim_qa.py.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

COUNT_WORDS = ("how many", "number of", "hvor mange", "mitu", "wie viele", "wieviele", "combien",
               "kuinka monta", "antal", "anzahl")

# (category, IFC classes, words). Order matters: first match wins.
CATEGORIES: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = [
    ("sanitary fixtures", ("IfcSanitaryTerminal",), ("sanitary", "sanitaer", "sanitar", "sanitaar", "wc ", "toilet")),
    ("stairs", ("IfcStair",), ("stair", "trappe", "trepp", "treppe", "escalier", "porra")),
    ("slabs", ("IfcSlab",), ("slab", "daek", "plade", "decke", "bodenplatte", "plaat", "dalle", "laatta")),
    ("doors", ("IfcDoor",), ("door", "dor", "ust", "uks", "tur", "porte", "ovi", "ovea", "ovet")),
    ("windows", ("IfcWindow",), ("window", "vindu", "aken", "akn", "fenster", "fenetre", "ikkun")),
    ("columns", ("IfcColumn",), ("column", "sojle", "post", "stutze", "saule", "poteau", "pilar")),
    ("beams", ("IfcBeam",), ("beam", "bjaelke", "tala", "trager", "poutre", "palk")),
    ("walls", ("IfcWall",), ("wall", "vaeg", "sein", "wand", "wande", "mur", "sein")),
    ("rooms", ("IfcSpace",), ("room", "space", "rum", "ruum", "raum", "raume", "piece", "tila", "huone")),
]
MEASURES = [  # (measure, quantity name in Qto sets, unit kind, words)
    ("length", "Length", "LENGTHUNIT", ("length", "laengde", "wie lang", "pikkus", "longueur", "pituus")),
    ("side area", "NetSideArea", "AREAUNIT", ("side area", "sideareal", "seitenflache", "kulgpind", "surface laterale")),
    ("volume", "NetVolume", "VOLUMEUNIT", ("volume", "volumen", "maht", "tilavuus")),
]
UNIT_LABEL = {"LENGTHUNIT": "m", "AREAUNIT": "m2", "VOLUMEUNIT": "m3"}


def _norm(text: Any) -> str:
    text = str(text or "").replace("æ", "ae").replace("Æ", "AE").replace("ø", "o").replace("Ø", "O")
    text = unicodedata.normalize("NFKD", text).casefold()
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _has_word(text: str, words: Sequence[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w.strip())}", text) for w in words)


# --------------------------------------------------------------------------- model index

class IfcIndex:
    """Lazily opened IFC model with storey lookup and unit scales."""

    def __init__(self, path: str):
        import ifcopenshell
        import ifcopenshell.util.unit as unit_util

        self.path = str(path)
        self.model = ifcopenshell.open(self.path)
        self.scale = {kind: unit_util.calculate_unit_scale(self.model, kind) for kind in UNIT_LABEL}
        self.storeys = {s.Name: s for s in self.model.by_type("IfcBuildingStorey") if s.Name}
        self._storey_cache: Dict[int, Optional[str]] = {}

    def storey_of(self, element: Any) -> Optional[str]:
        import ifcopenshell.util.element as E
        key = element.id()
        if key not in self._storey_cache:
            node = E.get_container(element) or E.get_aggregate(element)
            seen = 0
            while node is not None and not node.is_a("IfcBuildingStorey") and seen < 20:
                node = E.get_aggregate(node) or E.get_container(node)
                seen += 1
            self._storey_cache[key] = node.Name if node is not None and node.is_a("IfcBuildingStorey") else None
        return self._storey_cache[key]

    def elements(self, category: str) -> List[Any]:
        import ifcopenshell.util.element as E
        m = self.model
        if category == "sanitary fixtures":
            try:
                found = list(m.by_type("IfcSanitaryTerminal"))
            except RuntimeError:  # not in IFC2x3
                found = []
            if not found:
                found = [e for e in m.by_type("IfcFlowTerminal")
                         if (t := E.get_type(e)) is not None and t.is_a("IfcSanitaryTerminalType")]
            return found
        if category == "stairs":
            return [s for s in m.by_type("IfcStair")
                    if not ((parent := E.get_aggregate(s)) is not None and parent.is_a("IfcStair"))]
        classes = next(c for name, c, _ in CATEGORIES if name == category)
        return [e for cls in classes for e in m.by_type(cls)]

    def quantity(self, element: Any, name: str, unit_kind: str) -> Tuple[Optional[float], str]:
        import ifcopenshell.util.element as E
        for set_name, values in E.get_psets(element, qtos_only=True).items():
            value = values.get(name)
            if isinstance(value, (int, float)):
                return float(value) * self.scale[unit_kind], f"authored {set_name}.{name}"
        if unit_kind == "VOLUMEUNIT":
            try:
                import ifcopenshell.geom as geom
                import ifcopenshell.util.shape as shape_util
                settings = geom.settings()
                settings.set("use-world-coords", True)
                shape = geom.create_shape(settings, element)
                return float(shape_util.get_volume(shape.geometry)), "geometry volume (ifcopenshell)"
            except Exception:
                pass
        return None, "missing"


def _evidence(index: IfcIndex, elements: Sequence[Any], **extra: Any) -> Dict[str, Any]:
    return {"source": index.path, **extra,
            "elements": [{"guid": e.GlobalId, "ifc_type": e.is_a(), "name": e.Name,
                          "storey": index.storey_of(e)} for e in elements]}


def _not_found(reason: str) -> Dict[str, Any]:
    return {"status": "not_found", "value": None, "reason": reason, "evidence": []}


# --------------------------------------------------------------------------- answering

def classify(question: str) -> Optional[Tuple[str, Optional[str]]]:
    """Return (category, measure) if this is a count or wall/slab quantity question, else None."""
    q = _norm(question)
    category = next((name for name, _, words in CATEGORIES if _has_word(q, words)), None)
    if category is None:
        return None
    if _has_word(q, COUNT_WORDS):
        return category, None
    if category in ("walls", "slabs"):
        # Measure words are matched inside compounds (nettovolumen, Seitenflache).
        measure = next((m for m, _, _, words in MEASURES if any(w in q for w in words)), None)
        if measure:
            return category, measure
    return None


def answer_ifc_question(question: str, index: IfcIndex) -> Dict[str, Any]:
    kind = classify(question)
    if kind is None:
        return {"status": "unsupported", "reason": "Not an element count or wall/slab quantity question."}
    category, measure = kind

    # Scope: the longest storey name quoted in the question, else the whole model.
    storey = next((name for name in sorted(index.storeys, key=len, reverse=True)
                   if re.search(rf"[\'\"‘’“”„«»]\s*{re.escape(name)}\s*[\'\"‘’“”„«»]", question)), None)
    elements = index.elements(category)
    if storey is not None:
        elements = [e for e in elements if index.storey_of(e) == storey]
    scope = f"storey {storey!r}" if storey else "whole model"

    if measure is None:
        return {"status": "answered", "value": len(elements), "unit": "count",
                "method": f"{category} in {scope} (IFC entities)",
                "evidence": [_evidence(index, elements, scope=scope)]}

    _, qname, unit_kind, _ = next(m for m in MEASURES if m[0] == measure)
    if category == "slabs" and measure != "volume":
        return _not_found("Only slab volume is supported.")
    total, methods, missing = 0.0, set(), []
    for element in elements:
        value, method = index.quantity(element, qname, unit_kind)
        if value is None:
            missing.append(element.GlobalId)
            continue
        total += value
        methods.add(method)
    if not elements:
        return _not_found(f"No {category} found in {scope}.")
    if missing:
        return _not_found(f"{len(missing)} of {len(elements)} {category} have no {qname} quantity.")
    return {"status": "answered", "value": round(total, 3), "unit": UNIT_LABEL[unit_kind],
            "method": f"sum of {qname} over {len(elements)} {category} in {scope}; " + ", ".join(sorted(methods)),
            "evidence": [_evidence(index, elements, scope=scope)]}
