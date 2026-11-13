"""Question-to-evidence helpers for parsed IFC models (initially area Q&A)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _normalise(text: Any) -> str:
    text = unicodedata.normalize("NFKD", str(text or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text)


def _normalise_identifier(text: Any) -> str:
    text = unicodedata.normalize("NFKD", str(text or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9.-]+", "", text)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    # The source IFCs include decimal-comma properties, e.g. "63,8".
    if re.fullmatch(r"[+-]?\d+,\d+", text):
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _area_candidates(obj: Dict[str, Any]) -> List[Tuple[int, float, str, str, str]]:
    """Return (priority, value, unit, set name, property name) candidates."""
    candidates = []

    def add_entries(groups: Dict[str, Any], is_quantity: bool) -> None:
        for set_name, entries in groups.items():
            if not isinstance(entries, dict):
                continue
            for name, record in entries.items():
                if not isinstance(record, dict):
                    continue
                value = _number(record.get("value"))
                if value is None:
                    continue
                normalized = _normalise(name)
                quantity_type = _normalise(record.get("quantity_type", ""))
                if is_quantity and quantity_type and "area" not in quantity_type:
                    continue
                if any(term in normalized for term in ("grossfloorarea", "grossarea", "grosspindala")):
                    continue
                elif "netfloorarea" in normalized or "netarea" in normalized or "netsidearea" in normalized:
                    priority = 0
                elif "pindala" in normalized:
                    priority = 1
                elif "floorarea" in normalized or "floor_area" in name.casefold():
                    priority = 2
                elif is_quantity and "area" in quantity_type:
                    priority = 3
                elif "area" in normalized:
                    priority = 4
                else:
                    continue
                candidates.append((priority, value, str(record.get("unit") or ""), set_name, name))

    add_entries(obj.get("quantities", {}), True)
    add_entries(obj.get("properties", {}), False)
    # Geometry-derived fallback from bim_parser (used only when nothing authored exists).
    for name, record in (obj.get("computed_quantities") or {}).items():
        if isinstance(record, dict) and _number(record.get("value")) is not None:
            candidates.append((9, _number(record["value"]), "m2", "computed_geometry", name))
    candidates.sort(key=lambda item: item[0])
    positive = [candidate for candidate in candidates if candidate[1] > 0]
    return positive


def _space_name_text(obj: Dict[str, Any]) -> str:
    values = [obj.get("name"), obj.get("long_name"), obj.get("object_type")]
    for pset in obj.get("properties", {}).values():
        if isinstance(pset, dict):
            values.extend(item.get("value") for item in pset.values() if isinstance(item, dict))
    return " ".join(str(value) for value in values if value is not None)


def _matches_room_reference(target: str, obj: Dict[str, Any]) -> bool:
    target_norm = _normalise_identifier(target)
    if not target_norm:
        return False
    values = [obj.get("name"), obj.get("long_name"), obj.get("object_type")]
    for pset in obj.get("properties", {}).values():
        if isinstance(pset, dict):
            values.extend(item.get("value") for item in pset.values() if isinstance(item, dict))
    for value in values:
        if value is None:
            continue
        value_text = str(value)
        if _normalise_identifier(value_text) == target_norm:
            return True
        identifiers = re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", value_text.casefold())
        if any(_normalise_identifier(identifier) == target_norm for identifier in identifiers):
            return True
    return False


def _extract_room_reference(question: str) -> Optional[str]:
    # Challenge questions quote room labels such as "KORIDOR (119)" and
    # "Depot (Rum nr. S09)". Use the room number/code inside the parentheses.
    for match in re.finditer(r"\(([^()]*)\)", question):
        inside = match.group(1).strip()
        # Ignore clarifying phrases in aggregate questions such as
        # "all rooms (spaces)".
        if _normalise(inside) in {"room", "rooms", "space", "spaces", "m2", "m3"}:
            continue
        number = re.search(r"(?:rum\s*(?:nr\.?\s*)?|room\s*(?:no\.?\s*)?)([a-z0-9.-]+)", inside, re.I)
        return number.group(1) if number else inside
    return None


def _find_storey(question: str, model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qnorm = _normalise(question)
    matches = []
    for storey in model.get("storeys", []):
        name = str(storey.get("name") or "")
        normalized = _normalise(name)
        if normalized and normalized in qnorm:
            matches.append((len(normalized), storey))
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def _unit_to_m2(value: float, unit: str) -> Tuple[float, str]:
    normalized = _normalise(unit)
    if normalized in {"squaremillimetre", "squaremillimeter", "mm2", "millimetresquared", "millimetersquared"}:
        return value / 1_000_000, "m2"
    if normalized in {"squarecentimetre", "squarecentimeter", "cm2"}:
        return value / 10_000, "m2"
    if normalized in {"squarefoot", "squarefeet", "ft2"}:
        return value * 0.09290304, "m2"
    if normalized in {"squaremetre", "squaremeter", "m2"}:
        return value, "m2"
    # Dataset area questions request m²; IFC area values without explicit units
    # use the model's assigned area unit (all mapped challenge cases are m²).
    return value, "m2"


def answer_area_question(question: str, model: Dict[str, Any]) -> Dict[str, Any]:
    """Answer one room-area or storey-total-area question from ``parse_ifc`` data.

    Returns a structured answer with element-level provenance. If there is no
    sufficiently supported match, returns ``status='not_found'`` and an empty
    evidence list rather than guessing.
    """
    storey = _find_storey(question, model)
    storey_norm = _normalise(storey.get("name")) if storey else ""
    room_ref = _extract_room_reference(question)
    # The challenge JSONL contains some legacy-decoded Danish text ("v�gge").
    wall_area_question = any(token in _normalise(question) for token in ("wall", "walls", "vaeg", "vagg", "vgge"))
    if wall_area_question and not room_ref:
        spaces = [o for o in model.get("objects", []) if str(o.get("ifc_type", "")).startswith("IfcWall")]
    else:
        spaces = [o for o in model.get("objects", []) if o.get("ifc_type") == "IfcSpace"]
    if storey_norm:
        spaces = [o for o in spaces if _normalise((o.get("storey") or {}).get("name")) == storey_norm]

    def evidence_for(obj: Dict[str, Any], candidate: Tuple[int, float, str, str, str], value_m2: float) -> Dict[str, Any]:
        _, value, unit, set_name, property_name = candidate
        return {
            "value": value_m2,
            "unit": "m2",
            "element": {
                "ifc_type": obj.get("ifc_type"),
                "name": obj.get("name"),
                "long_name": obj.get("long_name"),
                "storey": (obj.get("storey") or {}).get("name"),
                "guid": obj.get("guid"),
                "ifc_id": (obj.get("source") or {}).get("ifc_id"),
            },
            "property": {"set": set_name, "name": property_name, "raw_value": value, "unit": unit or None},
            "source": (obj.get("source") or {}).get("model"),
        }

    matching_spaces = spaces
    if room_ref:
        target = _normalise(room_ref)
        direct = [o for o in spaces if _matches_room_reference(room_ref, {**o, "properties": {}})]
        matching_spaces = direct or [o for o in spaces if _matches_room_reference(room_ref, o)]
        if len(matching_spaces) > 1:
            # Prefer exact identifier matches in IfcSpace.Name.
            exact = [o for o in matching_spaces if _normalise(o.get("name")) == target]
            if exact:
                matching_spaces = exact
        if len(matching_spaces) == 1:
            obj = matching_spaces[0]
            candidates = _area_candidates(obj)
            if candidates:
                best = candidates[0]
                value, unit = _unit_to_m2(best[1], best[2])
                return {
                    "status": "answered", "value": value, "unit": unit,
                    "method": "room_property_or_quantity", "storey": storey.get("name") if storey else None,
                    "evidence": [evidence_for(obj, best, value)],
                }
        return {"status": "not_found", "reason": "Could not uniquely match the requested room and area property.", "evidence": []}

    # A storey total is requested when the question names a level but no room.
    # Sum one area per requested object and abstain if any object lacks a
    # usable nonzero area, rather than returning a partial total.
    evidence = []
    for obj in matching_spaces:
        candidates = _area_candidates(obj)
        if candidates:
            best = candidates[0]
            value, _ = _unit_to_m2(best[1], best[2])
            evidence.append(evidence_for(obj, best, value))
    if evidence and storey and len(evidence) == len(matching_spaces):
        total = sum(item["value"] for item in evidence)
        return {
            "status": "answered", "value": total, "unit": "m2",
            "method": "sum_element_areas", "storey": storey.get("name"),
            "evidence": evidence,
        }
    return {"status": "not_found", "reason": "No complete, supported area evidence found for the requested storey.", "evidence": []}


def score_numeric_answer(result: Dict[str, Any], expected: Any, tolerance: float = 0.02) -> Optional[bool]:
    """Compare a numeric answer using the challenge's 2% dimensional tolerance."""
    if result.get("status") != "answered" or not isinstance(expected, (int, float)):
        return None
    expected = float(expected)
    actual = float(result["value"])
    return abs(actual - expected) <= max(abs(expected) * tolerance, 0.001)
