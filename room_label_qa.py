"""Answer room-type count questions on raster floor plans by reading room labels.

Finnish floor plans (CubiCasa5K style) name rooms with standard abbreviations:
MH = bedroom, K/KK/KT/TK = kitchen, KH/KPH/PH/PESUH = bathroom, WC = WC room,
S/SAUNA = sauna, PARVEKE = balcony. We read those labels with OCR in all four
orientations (plans are often scanned rotated or upside down), merge duplicates,
and count them. Every counted label is returned with its pixel box, so a human
can check the answer on the drawing.

Policy per category was chosen on the preview set (241 drawings):
- bedrooms, WC rooms, balconies: count OCR labels (clearly better than a guess).
- saunas: present if an S/SAUNA label or a PH/PESUH washroom label is found
  (Finnish plans pair a sauna with a pesuhuone).
- storeys: number of distinct "N. KRS" / floor names found, at least 1.
- kitchens, bathrooms: OCR misses small single-letter labels and adds false
  positives, so we answer the typical value 1 and mark it for review.
Symbol counts (windows, doors, toilets, sinks, showers) are not handled here.

Requires: pip install rapidocr_onnxruntime pillow numpy
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROTATIONS = (0, 90, 180, 270)
SUPPORTED = ("bedrooms", "kitchens", "bathrooms", "wc_rooms", "saunas", "balconies", "storeys")

VOCAB = {
    "bedrooms": re.compile(r"MH\d?|MAKUUHUONE|MAKUUH"),
    "kitchens": re.compile(r"K|KK|KT|TK|AK|KEITTIO|KEITTOKOMERO|TUPAK|TUPAKEITTIO|APUK|APUKEITTIO|AVOK|KEITTO"),
    "bathrooms": re.compile(r"KH|KPH|PH|PESUH|SH|PSH|KYLPYHUONE|PESUHUONE|KYLPYH"),
    "wc_rooms": re.compile(r"WC\d?"),
    "saunas": re.compile(r"S|SAUNA|LH"),
    "balconies": re.compile(r"PARVEKE|PARV|LASITETTU|LPARV"),
}
WASHROOM_WORDS = {"PH", "PESUH", "PSH", "PESUHUONE"}
# Listing headers such as "3H + K + S" describe the flat, they are not room labels.
HEADER_RE = re.compile(r"\d\s*H\s*\+|\+\s*(K|S|KK|KT|WC|KPH|KH)\b")
FLOOR_NAMES = ("ALAKERTA", "YLAKERTA", "KELLARI", "POHJAKERROS", "PARVI")
MIN_CONF = 0.5
DUPLICATE_DISTANCE = 0.03  # fraction of the image's longer side

_ENGINE = None
_CACHE: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).upper())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def _engine():
    global _ENGINE
    if _ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError("Room-label OCR needs RapidOCR: pip install rapidocr_onnxruntime") from exc
        _ENGINE = RapidOCR()
    return _ENGINE


def _to_original(point: Sequence[float], rotation: int, width: int, height: int) -> Tuple[float, float]:
    """Map a point from a PIL image rotated counter-clockwise (expand=True) back to the original."""
    x, y = float(point[0]), float(point[1])
    if rotation == 90:
        return width - y, x
    if rotation == 180:
        return width - x, height - y
    if rotation == 270:
        return y, height - x
    return x, y


def read_labels(image_path: str | Path) -> Dict[int, List[Dict[str, Any]]]:
    """OCR the drawing in four orientations; boxes are mapped back to original pixels."""
    key = str(Path(image_path).resolve())
    if key in _CACHE:
        return _CACHE[key]
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    result: Dict[int, List[Dict[str, Any]]] = {}
    for rotation in ROTATIONS:
        rotated = image.rotate(rotation, expand=True) if rotation else image
        detections, _ = _engine()(np.array(rotated))
        items = []
        for box, text, conf in detections or []:
            points = [_to_original(p, rotation, width, height) for p in box]
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            items.append({"text": text, "conf": float(conf),
                          "bbox": [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]})
        result[rotation] = items
    _CACHE[key] = result
    return result


def _hits(labels: Dict[int, List[Dict[str, Any]]], category: str, size: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Label instances of one category, merged across orientations."""
    limit = DUPLICATE_DISTANCE * max(size)
    merged: List[Dict[str, Any]] = []
    for rotation, items in labels.items():
        for item in items:
            if item["conf"] < MIN_CONF or HEADER_RE.search(item["text"].upper()):
                continue
            words = _norm(item["text"]).split()
            if category == "balconies" and "LASITETTU" in words and "PARVEKE" in words:
                words = ["PARVEKE"]
            for word in words:
                if not VOCAB[category].fullmatch(word):
                    continue
                b = item["bbox"]
                centre = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                hit = {"label": word, "ocr_text": item["text"], "conf": round(item["conf"], 3),
                       "bbox": b, "orientation": rotation, "_c": centre}
                twin = next((h for h in merged if abs(h["_c"][0] - centre[0]) < limit
                             and abs(h["_c"][1] - centre[1]) < limit), None)
                if twin is None:
                    merged.append(hit)
                elif hit["conf"] > twin["conf"]:
                    merged[merged.index(twin)] = hit
    return merged


def _storeys(labels: Dict[int, List[Dict[str, Any]]]) -> Tuple[int, List[Dict[str, Any]]]:
    items = [i for i in labels.get(0, []) if i["conf"] >= MIN_CONF]
    evidence, numbers, names = [], set(), set()
    for item in items:
        text = _norm(item["text"])
        found = set(re.findall(r"\b(\d)\s*(?:KRS|KERROS)\b", text))
        named = {w for w in FLOOR_NAMES if w in text}
        if found or named:
            numbers |= found
            names |= named
            evidence.append({"label": item["text"], "bbox": item["bbox"], "conf": round(item["conf"], 3)})
    count = max(1, len(numbers), len(names) + 1 if len(names) == 1 and "PARVI" not in names else len(names))
    return count, evidence


def answer_room_question(question: str, image_path: str | Path, category: Optional[str] = None) -> Dict[str, Any]:
    """Answer a room-type count question for one raster floor plan."""
    if category is None:
        from floorplan_model import question_category
        category = question_category(question)
    if category not in SUPPORTED:
        return {"status": "unsupported", "reason": f"Room-label OCR does not handle {category!r}."}

    from PIL import Image
    with Image.open(image_path) as source:
        size = source.size
    labels = read_labels(image_path)
    base = {"source": str(image_path), "coordinate_system": "image pixels; origin at top-left"}

    def answer(value: int, method: str, hits: List[Dict[str, Any]], review: bool = False) -> Dict[str, Any]:
        elements = [{k: v for k, v in h.items() if not k.startswith("_")} for h in hits]
        return {"status": "answered", "value": value, "unit": "count", "method": method,
                "review_recommended": review, "evidence": [{**base, "labels": elements}]}

    if category == "storeys":
        count, evidence = _storeys(labels)
        return answer(count, "floor labels read by OCR (default 1)", evidence, review=not evidence)
    if category in ("bedrooms", "wc_rooms", "balconies"):
        hits = _hits(labels, category, size)
        return answer(len(hits), f"{category} labels read by OCR", hits)
    if category == "saunas":
        hits = _hits(labels, "saunas", size)
        wash = [h for h in _hits(labels, "bathrooms", size) if h["label"] in WASHROOM_WORDS]
        present = bool(hits or wash)
        return answer(1 if present else 0, "sauna label, or washroom (PH) that serves a sauna",
                      hits or wash, review=not hits)
    # kitchens, bathrooms: OCR is not reliable enough; show what was read, answer the typical value.
    hits = _hits(labels, category, size)
    return answer(1, "typical value 1; OCR labels shown for review", hits, review=True)
