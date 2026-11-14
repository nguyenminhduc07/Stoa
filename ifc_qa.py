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
        additional = {
            "coverings": ("IfcCovering",), "railings": ("IfcRailing",),
            "furniture": ("IfcFurnishingElement",), "members": ("IfcMember",),
            "plates": ("IfcPlate",), "curtain walls": ("IfcCurtainWall",),
            "pipes": ("IfcPipeSegment", "IfcPipeFitting"),
            "ducts": ("IfcDuctSegment", "IfcDuctFitting"),
            "mep components": ("IfcFlowTerminal", "IfcFlowSegment", "IfcFlowFitting",
                               "IfcFlowController", "IfcEnergyConversionDevice",
                               "IfcDistributionControlElement", "IfcElectricDistributionBoard",
                               "IfcElectricTimeControl", "IfcLightFixture", "IfcOutlet",
                               "IfcSwitchingDevice", "IfcSensor", "IfcActuator", "IfcAlarm",
                               "IfcSanitaryTerminal", "IfcPump", "IfcBoiler", "IfcFan",
                               "IfcAirTerminal", "IfcAirTerminalBox", "IfcProtectiveDevice"),
        }
        if category in additional:
            found = []
            for cls in additional[category]:
                try:
                    found.extend(m.by_type(cls))
                except RuntimeError:
                    continue
            return list({element.id(): element for element in found}.values())
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


# --------------------------------------------------------------------------- IFC-Bench text questions

def _psets(element: Any) -> Dict[str, Dict[str, Any]]:
    import ifcopenshell.util.element as E
    try:
        return E.get_psets(element, should_inherit=True)
    except TypeError:
        return E.get_psets(element)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return value.wrappedValue
    except AttributeError:
        return str(value)


def _named(element: Any) -> str:
    for key in ("LongName", "Name", "ObjectType", "Tag"):
        value = getattr(element, key, None)
        if value:
            return str(value)
    return f"{element.is_a()} #{element.id()}"


def _type_name(element: Any) -> str:
    import ifcopenshell.util.element as E
    type_object = E.get_type(element)
    if type_object is not None:
        return str(getattr(type_object, "Name", None) or getattr(type_object, "ElementType", None) or type_object.is_a())
    return str(getattr(element, "ObjectType", None) or getattr(element, "Name", None) or element.is_a())


def _prop_value(index: IfcIndex, element: Any, names: Sequence[str], unit_kind: Optional[str] = None) -> Optional[float]:
    wanted = {_norm(name).replace(" ", "") for name in names}
    for values in _psets(element).values():
        for key, raw in values.items():
            if _norm(key).replace(" ", "") not in wanted:
                continue
            raw = _plain(raw)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                scale = index.scale.get(unit_kind, 1.0) if unit_kind else 1.0
                return float(raw) * scale
    return None


def _prop_raw(element: Any, names: Sequence[str]) -> Any:
    wanted = {_norm(name).replace(" ", "") for name in names}
    for values in _psets(element).values():
        for key, raw in values.items():
            if _norm(key).replace(" ", "") in wanted:
                return _plain(raw)
    return None


def _is_external(element: Any) -> Optional[bool]:
    value = getattr(element, "IsExternal", None)
    if value is None:
        value = _prop_raw(element, ("IsExternal", "External", "IsExterior"))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.casefold()
        if lowered in {"true", "yes", "1", "external", "exterior"}:
            return True
        if lowered in {"false", "no", "0", "internal", "interior"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _space_area(index: IfcIndex, element: Any, *, gross: bool = False) -> Optional[float]:
    names = ("GrossFloorArea", "GrossArea", "GrossFootprintArea") if gross else ("NetFloorArea", "NetArea", "NetFootprintArea", "GSA BIM Area")
    for name in names:
        value, _method = index.quantity(element, name, "AREAUNIT")
        if value is not None:
            return value
    return _prop_value(index, element, names, "AREAUNIT")


def _width(index: IfcIndex, element: Any) -> Optional[float]:
    for name in ("OverallWidth", "ClearWidth", "NominalWidth", "Width"):
        raw = getattr(element, name, None)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw) * index.scale.get("LENGTHUNIT", 1.0)
    value = _prop_value(index, element, ("OverallWidth", "ClearWidth", "NominalWidth", "Width"), "LENGTHUNIT")
    if value is not None:
        return value
    value, _method = index.quantity(element, "Width", "LENGTHUNIT")
    return value


def _window_area_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    if "window" not in q and "glazed" not in q:
        return None
    if "area" not in q and "glazed" not in q:
        return None
    windows = list(index.model.by_type("IfcWindow"))
    if not windows:
        return _not_found("No IfcWindow elements are defined in the model.")
    values, estimated = [], False
    for window in windows:
        area = _prop_value(index, window, ("Area", "GrossArea", "OverallArea"), "AREAUNIT")
        if area is None and "estimated" in q:
            width = _width(index, window)
            height = None
            for attr in ("OverallHeight", "Height"):
                raw = getattr(window, attr, None)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    height = float(raw) * index.scale.get("LENGTHUNIT", 1.0)
                    break
            if height is None:
                height = _prop_value(index, window, ("OverallHeight", "Height"), "LENGTHUNIT")
            if width is not None and height is not None:
                area = width * height
                estimated = True
        if area is not None:
            values.append((window, area))
    if not values:
        return _not_found("No authored window area or complete width/height dimensions were found.")
    total = sum(area for _, area in values)
    coverage = f"for {len(values)} of {len(windows)} windows"
    if estimated:
        lead = f"Estimated gross window opening area (OverallWidth x OverallHeight): {total:.2f} m2 {coverage}. This is not net glass area."
        method = "estimated from authored IFC window overall dimensions"
    else:
        lead = f"Total authored window area: {total:.2f} m2 {coverage}."
        method = "sum of authored IFC window area properties"
    return _text_answer(index, lead, method, [window for window, _ in values])


def _text_evidence(index: IfcIndex, elements: Sequence[Any], **extra: Any) -> List[Dict[str, Any]]:
    return [_evidence(index, elements, **extra)] if elements else [{"source": index.path, **extra}]


def _text_answer(index: IfcIndex, answer: str, method: str, elements: Sequence[Any] = (), **extra: Any) -> Dict[str, Any]:
    return {"status": "answered", "answer": answer, "method": method,
            "evidence": _text_evidence(index, elements, **extra)}


def _find_question_storey(question: str, storeys: Sequence[Any], index: Optional[IfcIndex] = None) -> Optional[Any]:
    q = _norm(question)
    matches = []
    for storey in storeys:
        aliases = (getattr(storey, "Name", None), getattr(storey, "LongName", None),
                   getattr(storey, "Description", None))
        if any(alias and _norm(alias) in q for alias in aliases):
            matches.append(storey)
    exact = max(matches, key=lambda item: max(len(_norm(getattr(item, key, "") or ""))
                                               for key in ("Name", "LongName", "Description")), default=None)
    if exact is not None:
        return exact

    ordered = sorted(storeys, key=lambda item: (
        getattr(item, "Elevation", None) is None,
        getattr(item, "Elevation", 0) or 0,
        item.id(),
    ))
    if index is not None:
        with_spaces = [item for item in ordered if _storey_spaces(index, item)]
        if with_spaces:
            ordered = with_spaces
    ground_index = next((i for i, item in enumerate(ordered)
                         if "ground" in _norm(getattr(item, "Description", ""))
                         or "ground" in _norm(getattr(item, "Name", ""))),
                        next((i for i, item in enumerate(ordered)
                              if (getattr(item, "Elevation", None) or 0) >= 0), 0))
    cues = (
        (("basement", "cellar", "underground", "lower ground", "lower level", "kjeller", "keller"),
         lambda values: min(values, key=lambda item: (getattr(item, "Elevation", 0) or 0, item.id()))),
        (("ground floor", "ground level", "ground story"), lambda values: values[ground_index]),
        (("second floor", "second level", "level 2"),
         lambda values: values[min(ground_index + 2, len(values) - 1)]),
        (("first floor", "first level", "level 1", "1st floor", "first storey"),
         lambda values: values[min(ground_index + 1, len(values) - 1)]),
        (("attic", "roof level", "top floor", "uppermost floor"), lambda values: values[-1]),
    )
    for words, choose in cues:
        if any(word in q for word in words) and ordered:
            return choose(ordered)
    level_match = re.search(r"\blevel\s+([a-z]?\d{1,2})\b", q)
    if level_match:
        token = _norm(level_match.group(1))
        for storey in storeys:
            if token and token in _norm(getattr(storey, "Name", "")):
                return storey
    return None


def _storey_spaces(index: IfcIndex, storey: Any) -> List[Any]:
    return [space for space in index.model.by_type("IfcSpace") if index.storey_of(space) == storey.Name]


def _space_breakdown(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    model = index.model
    storeys = list(model.by_type("IfcBuildingStorey"))
    spaces = list(model.by_type("IfcSpace"))
    if not spaces:
        return _not_found("The IFC model contains no IfcSpace elements to list.")
    requested = _find_question_storey(question, storeys, index)
    selected_storeys = [requested] if requested else sorted(
        storeys, key=lambda s: (getattr(s, "Elevation", None) is None, getattr(s, "Elevation", 0) or 0)
    )
    if requested is None and any(word in _norm(question) for word in ("on the basement", "on basement", "underground level", "on the ground", "on level", "on floor")):
        return _not_found("Could not match the requested floor name to an IfcBuildingStorey; no all-floor result was substituted.")
    if requested is not None and not _storey_spaces(index, requested):
        return _not_found(f"No IfcSpace elements are assigned to the requested storey {_named(requested)}.")
    output = []
    cited = []
    for storey in selected_storeys:
        floor_spaces = _storey_spaces(index, storey)
        cited.extend(floor_spaces)
        items = []
        total = 0.0
        complete = True
        for space in floor_spaces:
            area = _space_area(index, space)
            if area is None:
                complete = False
                items.append(f"- {_named(space)}: area not authored")
            else:
                total += area
                items.append(f"- {_named(space)}: {area:.2f} m2")
        floor_name = str(getattr(storey, "Name", None) or "Unnamed storey")
        floor_total = f"{total:.2f} m2" if complete else f"at least {total:.2f} m2 (some spaces have no area)"
        output.append(f"{floor_name}: {len(floor_spaces)} spaces; total {floor_total}")
        output.extend(items)
    if not selected_storeys:
        return _not_found("The model has spaces but no named building storeys.")
    answer = "\n".join(output)
    return _text_answer(index, answer, "IFC IfcSpace quantities grouped by storey", cited)


def _level_heights(index: IfcIndex) -> Optional[Dict[str, Any]]:
    levels = [s for s in index.model.by_type("IfcBuildingStorey") if getattr(s, "Elevation", None) is not None]
    levels.sort(key=lambda s: float(s.Elevation))
    if len(levels) < 2:
        return None
    scale = index.scale.get("LENGTHUNIT", 1.0)
    rows = []
    for position, storey in enumerate(levels):
        elevation = float(storey.Elevation) * scale
        authored_height = _prop_value(index, storey, ("Height", "GrossHeight", "NetHeight"), "LENGTHUNIT")
        if authored_height is None and position + 1 < len(levels):
            authored_height = (float(levels[position + 1].Elevation) - float(storey.Elevation)) * scale
        height_text = f"; height {authored_height:.2f} m" if authored_height is not None else ""
        rows.append(f"{_named(storey)}: elevation {elevation:.2f} m{height_text}")
    answer = "IfcBuildingStorey elevations and authored/heights between levels:\n" + "\n".join(rows)
    return _text_answer(index, answer, "IfcBuildingStorey.Elevation differences", levels)


def _space_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    spaces = list(index.model.by_type("IfcSpace"))
    if not spaces:
        return _not_found("No IfcSpace elements are defined in this model.")
    if ("largest" in q or "biggest" in q) and ("room" in q or "space" in q):
        values = [(area, space) for space in spaces if (area := _space_area(index, space)) is not None]
        if not values:
            return _not_found("No authored room area was found, so the largest space cannot be identified.")
        area, space = max(values, key=lambda pair: pair[0])
        return _text_answer(index, f"Largest space: {_named(space)}, {area:.2f} m2.",
                            "largest IfcSpace by authored net floor area", [space])
    if any(term in q for term in ("space breakdown", "all spaces", "all the spaces", "each room", "room list", "room breakdown", "spaces on", "spaces are on", "rooms on", "space distribution", "spaces by level", "space by level", "all rooms", "all the rooms", "names of all rooms", "rooms in the building")):
        return _space_breakdown(index, question)
    if any(term in q for term in ("space utilization", "space used by function", "space utilized by function", "area distribution by function")):
        groups: Dict[str, List[Tuple[Any, float]]] = {}
        for space in spaces:
            area = _space_area(index, space)
            if area is None:
                continue
            function = str(getattr(space, "ObjectType", None) or getattr(space, "LongName", None)
                           or getattr(space, "Name", None) or "Unclassified")
            groups.setdefault(function, []).append((space, area))
        if not groups:
            return _not_found("No authored IfcSpace areas were found for a function-area breakdown.")
        total = sum(area for items in groups.values() for _, area in items)
        rows = [f"{name}: {sum(area for _, area in items):.2f} m2 ({100 * sum(area for _, area in items) / total:.1f}%), {len(items)} spaces"
                for name, items in sorted(groups.items(), key=lambda pair: -sum(area for _, area in pair[1]))]
        cited = [space for items in groups.values() for space, _ in items]
        return _text_answer(index, f"Space area utilization by authored function (total {total:.2f} m2):\n" + "\n".join(rows),
                            "IfcSpace authored function grouped by net floor area", cited)
    room_function = next((name for name, aliases in (
        ("bedrooms", ("bedroom", "sleeping room")),
        ("bathrooms", ("bathroom", "bath room", "bathroom spaces")),
    ) if any(alias in q for alias in aliases)), None)
    if room_function and any(term in q for term in ("how many", "number of", "count", "are there")):
        matches = [space for space in spaces if any(alias in _norm(_named(space)) for alias in
                   (("bedroom", "sleeping room") if room_function == "bedrooms" else ("bathroom", "bath room")))]
        return _text_answer(index, f"{len(matches)} {room_function} identified from IfcSpace names.",
                            "IfcSpace name and LongName classification", matches)
    if "toilet" in q or "restroom" in q or "wc" in q:
        requested = _find_question_storey(question, list(index.model.by_type("IfcBuildingStorey")), index)
        if requested is not None and ("how many" in q or "number of" in q or "toilet spaces" in q or "restroom spaces" in q):
            wc_spaces = [space for space in _storey_spaces(index, requested)
                         if any(word in _norm(" ".join(str(getattr(space, key, "") or "") for key in ("Name", "LongName", "ObjectType")))
                                for word in ("toilet", "restroom", "washroom", "wc"))]
            return _text_answer(index, f"{len(wc_spaces)} toilet/restroom spaces on {_named(requested)}.",
                                "IfcSpace labels filtered by storey", wc_spaces)
    if ("by level" in q or "by floor" in q or "breakdown by level" in q or "each floor" in q or "each level" in q) and any(term in q for term in ("floor area", "usable area", "net area", "gross area", "area of each floor")):
        storeys = sorted(index.model.by_type("IfcBuildingStorey"),
                         key=lambda item: (getattr(item, "Elevation", None) is None, getattr(item, "Elevation", 0) or 0))
        rows = []
        cited = []
        complete = True
        total = 0.0
        for storey in storeys:
            floor_spaces = _storey_spaces(index, storey)
            quantity_names = ("GrossFloorArea", "GrossArea") if "gross" in q else ("NetFloorArea", "NetArea")
            area_sum = None
            for quantity_name in quantity_names:
                area_sum, _method = index.quantity(storey, quantity_name, "AREAUNIT")
                if area_sum is None:
                    area_sum = _prop_value(index, storey, (quantity_name,), "AREAUNIT")
                if area_sum is not None:
                    break
            floor_areas = [(space, _space_area(index, space, gross="gross" in q)) for space in floor_spaces]
            if area_sum is None and floor_areas and all(area is not None for _, area in floor_areas):
                area_sum = sum(area for _, area in floor_areas if area is not None)
            if area_sum is None:
                complete = False
                rows.append(f"{_named(storey)}: area not authored ({len(floor_spaces)} assigned spaces)")
                cited.extend(floor_spaces)
                continue
            total += area_sum
            rows.append(f"{_named(storey)}: {area_sum:.2f} m2 ({len(floor_spaces)} spaces)")
            cited.extend(floor_spaces)
        if rows:
            if total <= 0:
                return _not_found("The model has no authored per-storey or per-space floor areas for this request.")
            prefix = "Total" if complete else "Known partial"
            lead = f"{prefix} {'gross' if 'gross' in q else 'net'} floor area: {total:.2f} m2" + (" (some storeys have no authored area)" if not complete else "")
            return _text_answer(index, lead + "\n" + "\n".join(rows),
                                "sum of authored IfcSpace areas by IfcBuildingStorey", cited)
    if "total floor area" in q or "gross floor area" in q or "gfa" in q or "ufa" in q or "usable floor area" in q:
        gross = any(term in q for term in ("gross", "gfa"))
        areas = [(space, _space_area(index, space, gross=gross)) for space in spaces]
        known = [(space, area) for space, area in areas if area is not None]
        if known and len(known) == len(spaces):
            total = sum(area for _, area in known)
            cited = [space for space, _ in known]
            label = "gross" if gross else "net"
            return _text_answer(index, f"Total {label} area summed from {len(known)} IfcSpace elements: {total:.2f} m2.",
                                f"sum of authored {label} IfcSpace areas", cited)
        if known:
            return _not_found(f"Only {len(known)} of {len(spaces)} spaces have an authored {('gross' if gross else 'net')} area; refusing to report an incomplete total.")
        return _not_found("The model has no usable authored space-area quantities.")
    return None


def _type_breakdown(index: IfcIndex, category: str, question: str = "") -> Optional[Dict[str, Any]]:
    elements = index.elements(category)
    q = _norm(question)
    if category == "walls" and any(term in q for term in ("exterior", "external", "interior", "internal", "partition")):
        want_external = any(term in q for term in ("exterior", "external"))
        selected = [element for element in elements if _is_external(element) is want_external]
        elements = selected
    if not elements:
        return _not_found(f"No {category} are defined in this IFC model.")
    groups: Dict[str, List[Any]] = {}
    for element in elements:
        groups.setdefault(_type_name(element), []).append(element)
    scope = "exterior " if category == "walls" and any(term in q for term in ("exterior", "external")) else (
        "interior " if category == "walls" and any(term in q for term in ("interior", "internal", "partition")) else "")
    lines = [f"Total {scope}{category}: {len(elements)}"]
    for name, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0].casefold())):
        dimensions = []
        for dimension, aliases in (("width", ("OverallWidth", "Width", "ProfileWidth")),
                                  ("height", ("OverallHeight", "Height", "ProfileDepth"))):
            vals = [_prop_value(index, item, aliases, "LENGTHUNIT") for item in members]
            vals = [v for v in vals if v is not None]
            if vals:
                dimensions.append(f"{dimension} {sum(vals)/len(vals):.2f} m average")
        layers = _materials(members[0]) if category in {"walls", "slabs"} else []
        if layers:
            layer_text = ", ".join(
                f"{mat} ({thickness * 1000:.0f} mm)" if thickness is not None else mat
                for mat, thickness in layers
            )
            dimensions.append(f"materials/layers: {layer_text}")
        suffix = f" ({'; '.join(dimensions)})" if dimensions else ""
        percentage = 100.0 * len(members) / len(elements)
        lines.append(f"- {name}: {len(members)} ({percentage:.1f}%){suffix}")
    return _text_answer(index, "\n".join(lines), f"Ifc{category.title()} grouped by assigned type",
                        elements)


def _width_summary(index: IfcIndex, category: str, question: str) -> Optional[Dict[str, Any]]:
    elements = index.elements(category)
    widths = [(value, element) for element in elements if (value := _width(index, element)) is not None]
    if not widths:
        return _not_found(f"No authored width properties were found for {category}.")
    q = _norm(question)
    selected = []
    if any(word in q for word in ("average", "mean")):
        value = sum(width for width, _ in widths) / len(widths)
        cited = [e for _, e in widths]
        label = "Average"
    elif any(word in q for word in ("narrowest", "smallest", "minimum", "shortest")):
        value = min(width for width, _ in widths)
        selected = [e for width, e in widths if abs(width - value) < 1e-6]
        cited, label = selected, "Minimum"
    elif any(word in q for word in ("two", "2 ", "top", "largest", "greatest", "maximum", "widest")):
        limit_match = re.search(r"\b(\d+)\b", q)
        words_to_numbers = {"two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
        word_limit = next((number for word, number in words_to_numbers.items()
                           if re.search(rf"\b{word}\b", q)), None)
        limit = int(limit_match.group(1)) if limit_match else word_limit or 5
        selected = sorted(widths, key=lambda pair: pair[0], reverse=True)[:limit]
        cited = [e for _, e in selected]
        by_type = "type" in q or "types" in q
        rows = [f"{i}. {_type_name(element) if by_type else _named(element)}: {width:.3f} m"
                for i, (width, element) in enumerate(selected, 1)]
        return _text_answer(index, "Largest authored widths:\n" + "\n".join(rows),
                            f"top {len(selected)} {category} widths from IFC properties", cited)
    else:
        value = max(width for width, _ in widths)
        selected = [e for width, e in widths if abs(width - value) < 1e-6]
        cited, label = selected, "Maximum"
    element_name = _type_name(selected[0]) if selected and "type" in q else _named(selected[0]) if selected else ""
    answer = f"{label} authored width: {value:.3f} m" + (f" ({element_name})" if element_name else ".")
    return _text_answer(index, answer,
                        f"{label.lower()} {category} width from IFC properties", cited)


def _materials(element: Any) -> List[Tuple[str, Optional[float]]]:
    import ifcopenshell.util.element as E
    try:
        material = E.get_material(element, should_inherit=True)
    except Exception:
        material = None
    if material is None:
        return []
    if material.is_a("IfcMaterial"):
        return [(str(material.Name or "Unnamed material"), None)]
    material_set = getattr(material, "ForLayerSet", None) or material
    layers = getattr(material_set, "MaterialLayers", None) or []
    scale = 1.0
    try:
        scale = __import__("ifcopenshell.util.unit", fromlist=["calculate_unit_scale"]).calculate_unit_scale(element.file, "LENGTHUNIT")
    except Exception:
        pass
    result = []
    for layer in layers:
        mat = getattr(layer, "Material", None)
        name = str(getattr(mat, "Name", None) or "Unnamed layer")
        thickness = getattr(layer, "LayerThickness", None)
        result.append((name, float(thickness) * scale if isinstance(thickness, (int, float)) else None))
    return result


def _wall_types(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    has_material = "material" in q or "composition" in q or "layer" in q
    has_wall_scope = "wall" in q or "walls" in q
    if not ("wall type" in q or "types of wall" in q or "type distribution" in q and has_wall_scope
            or has_wall_scope and (has_material or any(word in q for word in ("breakdown", "distribution", "quantity", "quantities")))
            or "insulation" in q or "thickness" in q and has_wall_scope):
        return None
    walls = list(index.model.by_type("IfcWall")) + list(index.model.by_type("IfcWallStandardCase"))
    # IFC4 may return StandardCase through IfcWall; deduplicate by entity id.
    walls = list({wall.id(): wall for wall in walls}.values())
    if not walls:
        return _not_found("No IfcWall elements are defined in the model.")
    groups: Dict[str, List[Any]] = {}
    for wall in walls:
        groups.setdefault(_type_name(wall), []).append(wall)
    rows = []
    cited = []
    for type_name, members in sorted(groups.items()):
        if "exterior" in q or "external" in q:
            members = [item for item in members if _is_external(item) is True]
        elif "interior" in q or "internal" in q or "partition" in q:
            members = [item for item in members if _is_external(item) is False]
        if not members:
            continue
        layers = _materials(members[0])
        layer_text = "; ".join(f"{name} ({thickness:.3f} m)" if thickness is not None else name for name, thickness in layers)
        rows.append(f"- {type_name}: {len(members)} walls" + (f"; layers: {layer_text}" if layer_text else ""))
        cited.extend(members)
    if not rows:
        return _not_found("The IFC model does not author IsExternal values for the requested wall scope.")
    scope = "exterior " if "exterior" in q or "external" in q else (
        "interior " if "interior" in q or "internal" in q or "partition" in q else "")
    total = sum(len(items) for items in groups.values())
    if scope:
        total = len(cited)
    return _text_answer(index, f"Total {scope}walls: {total}\nWall types and available material layers:\n" + "\n".join(rows),
                        "IfcWall assigned types and associated material layer sets", cited)


def _project_metadata(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    model = index.model
    asks_software = any(term in q for term in ("software", "modelling program", "modeling program", "originating application", "ifc standard", "ifc schema", "modeling standards"))
    if asks_software:
        header = getattr(getattr(model, "header", None), "file_name", None)
        schema = getattr(model, "schema", None)
        originating = getattr(header, "originating_system", None)
        preprocessor = getattr(header, "preprocessor_version", None)
        answer = f"IFC schema: {schema or 'not recorded'}"
        if originating:
            answer += f". Originating software: {originating}."
        if preprocessor:
            answer += f" Preprocessor: {preprocessor}."
        return _text_answer(index, answer, "IFC file header and schema", [])

    if any(term in q for term in ("project information", "project name", "building name", "location", "coordinates", "building type", "building function", "building usage", "what type of building", "situated", "type and its function")):
        project = next(iter(model.by_type("IfcProject")), None)
        building = next(iter(model.by_type("IfcBuilding")), None)
        site = next(iter(model.by_type("IfcSite")), None)
        values = []
        cited = []
        for label, item, fields in (
            ("Project", project, ("Name", "LongName", "Description")),
            ("Building", building, ("Name", "LongName", "Description", "ObjectType", "CompositionType")),
            ("Site", site, ("Name", "LongName", "Description", "RefLatitude", "RefLongitude", "RefElevation")),
        ):
            if item is None:
                continue
            bits = [f"{field}: {_plain(getattr(item, field, None))}" for field in fields if getattr(item, field, None) not in (None, "")]
            if bits:
                values.append(f"{label}: " + "; ".join(bits))
                cited.append(item)
        if building is not None and any(term in q for term in ("building type", "building function", "building usage", "type of building", "type and its function")):
            type_values = []
            for key in ("BuildingType", "BuildingFunction", "BuildingUse", "OccupancyType", "Category"):
                value = _prop_raw(building, (key,))
                if value not in (None, ""):
                    type_values.append(f"{key}: {value}")
            if not type_values:
                name_hint = _norm(_named(building))
                if "buero" in name_hint or "office" in name_hint:
                    type_values.append("Possible office use (inferred from the building name; not an authored classification)")
            if type_values:
                values.append("Building function/use: " + "; ".join(type_values))
        if values:
            return _text_answer(index, "\n".join(values), "IfcProject/IfcBuilding/IfcSite authored attributes", cited)
        return _not_found("Project, building, and site metadata are not authored in this IFC model.")
    return None


def _building_scale(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    if not ("building footprint" in q or "footprint area" in q or
            ("number of levels" in q and ("floor area" in q or "spaces" in q))):
        return None
    building = next(iter(index.model.by_type("IfcBuilding")), None)
    storeys = list(index.model.by_type("IfcBuildingStorey"))
    spaces = list(index.model.by_type("IfcSpace"))
    rows = [f"Number of building storeys: {len(storeys)}", f"Total IfcSpace elements: {len(spaces)}"]
    cited: List[Any] = list(storeys) + list(spaces)
    footprint = None
    if building is not None:
        for name in ("GrossFootprintArea", "NetFootprintArea", "FootprintArea", "BuildingFootprintArea"):
            footprint, _method = index.quantity(building, name, "AREAUNIT")
            if footprint is None:
                footprint = _prop_value(index, building, (name,), "AREAUNIT")
            if footprint is not None:
                rows.append(f"Building footprint area ({name}): {footprint:.2f} m2")
                cited.append(building)
                break
    if footprint is None:
        rows.append("Building footprint area: not authored in the IFC model")
    gross_area = None
    if building is not None:
        gross_area, _method = index.quantity(building, "GrossFloorArea", "AREAUNIT")
        if gross_area is None:
            gross_area = _prop_value(index, building, ("GrossFloorArea",), "AREAUNIT")
    net_values = [(space, _space_area(index, space)) for space in spaces]
    known_net = [(space, area) for space, area in net_values if area is not None]
    if gross_area is not None:
        rows.append(f"Building GrossFloorArea: {gross_area:.2f} m2")
    if known_net and len(known_net) == len(spaces):
        net_area = sum(area for _, area in known_net)
        rows.append(f"Total net area summed from IfcSpace: {net_area:.2f} m2")
    elif known_net:
        rows.append(f"Known net area: {sum(area for _, area in known_net):.2f} m2 ({len(known_net)}/{len(spaces)} spaces have area)")
    return _text_answer(index, "Building scale from authored IFC data:\n" + "\n".join(rows),
                        "IfcBuilding properties, IfcBuildingStoreys and IfcSpace counts/quantities", cited)


def _generic_property_answer(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    requested = []
    if any(term in q for term in ("fire rating", "fire resistance")):
        requested = ["firerating", "fire resistance"]
    elif "glazing" in q:
        requested = ["glazing", "glass type"]
    elif "cost" in q or "price" in q:
        requested = ["cost", "price"]
    elif "capacity" in q:
        requested = ["capacity", "occupancy"]
    elif "voltage" in q:
        requested = ["voltage", "rated voltage"]
    elif "energy efficiency" in q or "efficiency rating" in q:
        requested = ["energy efficiency rating", "efficiency rating"]
    elif "maintenance" in q:
        requested = ["maintenance requirement", "maintenance interval"]
    elif "warranty" in q or "lifespan" in q or "expected life" in q:
        requested = ["warranty period", "service life", "expected life"]
    elif "depth" in q or "foundation" in q:
        requested = ["depth", "foundation depth", "embedment"]
    elif "air tightness" in q or "airtightness" in q:
        requested = ["air tightness", "airtightness"]
    elif "solar heat gain" in q or "shgc" in q:
        requested = ["solar heat gain coefficient", "shgc"]
    elif "accessibility" in q or "accessible" in q or "compliant" in q:
        requested = ["accessibility", "accessible", "compliance"]
    elif any(term in q for term in ("fire exit", "fire rated", "emergency exit", "escape route", "egress")):
        requested = ["fire exit", "fire rating", "escape route", "egress", "exit distance"]
    elif "recyclability" in q or "recyclable" in q:
        requested = ["recyclability", "recycled content", "end of life"]
    elif "electrical load" in q or "electric load" in q:
        requested = ["electrical load", "load", "power"]
    elif "u value" in q or "thermal" in q:
        requested = ["uvalue", "thermaltransmittance"]
    elif "ceiling height" in q or "room height" in q:
        requested = ["ceilingheight", "height"]
    if not requested:
        return None
    if "door" in q:
        elements = list(index.model.by_type("IfcDoor"))
    elif "window" in q or "glazing" in q:
        elements = list(index.model.by_type("IfcWindow"))
    elif "foundation" in q:
        elements = []
        for cls in ("IfcFooting", "IfcPile", "IfcSlab"):
            try:
                elements.extend(index.model.by_type(cls))
            except RuntimeError:
                pass
    elif "wall" in q or "insulation" in q or "thermal" in q:
        elements = list(index.model.by_type("IfcWall")) + list(index.model.by_type("IfcWallStandardCase"))
    elif "room" in q or "space" in q or "ceiling" in q:
        elements = list(index.model.by_type("IfcSpace"))
    else:
        elements = list(index.model.by_type("IfcProduct"))
    values = []
    cited = []
    for element in elements:
        for pset_name, pset in _psets(element).items():
            for name, value in pset.items():
                normalized = _norm(name).replace(" ", "")
                if any(term.replace(" ", "") in normalized for term in requested) and isinstance(_plain(value), (str, int, float)):
                    values.append(f"{_named(element)} ({element.is_a()}): {name} = {_plain(value)}")
                    cited.append(element)
    if values:
        # Keep evidence manageable but deterministic.
        unique = list(dict.fromkeys(values))
        return _text_answer(index, "\n".join(unique[:40]), "matching authored IFC property values", cited[:40])
    if requested:
        scope = "doors" if "door" in q else "windows" if "window" in q or "glazing" in q else "walls" if "wall" in q else "spaces" if "room" in q or "space" in q else "building products"
        return _text_answer(index, f"No {', '.join(requested)} property was found on the {scope} searched in this IFC model.",
                            "property-set search found no requested authored value", elements[:40])
    return _not_found(f"No matching authored properties ({', '.join(requested)}) were found.")


def _ceiling_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    if (not any(term in q for term in ("ceiling", "room height", "space height"))
            or "material" in q):
        return None
    spaces = list(index.model.by_type("IfcSpace"))
    values = [(height, space) for space in spaces
              if (height := _prop_value(index, space,
                                        ("FinishCeilingHeight", "ClearHeight", "CeilingHeight", "Height"),
                                        "LENGTHUNIT")) is not None]
    if not values:
        return _not_found("No authored ceiling/space height properties were found.")
    descending = any(term in q for term in ("tallest", "highest", "maximum", "highest ceiling"))
    ascending = any(term in q for term in ("shortest", "lowest", "minimum", "lowest ceiling"))
    if descending or ascending:
        limit_match = re.search(r"\btop\s+(\d+)\b", q)
        limit = min(50, int(limit_match.group(1))) if limit_match else 10
        selected = sorted(values, key=lambda pair: pair[0], reverse=descending)[:limit]
        label = "Highest" if descending else "Lowest"
    else:
        selected = sorted(values, key=lambda pair: (index.storey_of(pair[1]) or "", _named(pair[1]).casefold()))
        label = "Authored"
    rows = [f"{i}. {_named(space)}: {height:.2f} m (storey: {index.storey_of(space) or 'not assigned'})"
            for i, (height, space) in enumerate(selected, 1)]
    return _text_answer(index, f"{label} authored space/ceiling heights:\n" + "\n".join(rows),
                        "IfcSpace authored height properties sorted as requested",
                        [space for _, space in selected])


def _structural_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    if not any(term in q for term in ("structural element", "structural component", "structural system", "structural member", "roof structure", "foundation system")):
        return None
    categories = {
        "beams": ("IfcBeam",), "columns": ("IfcColumn",), "members": ("IfcMember",),
        "plates": ("IfcPlate",), "walls": ("IfcWall",), "slabs": ("IfcSlab",),
        "footings": ("IfcFooting",), "piles": ("IfcPile",), "roof elements": ("IfcRoof",),
    }
    rows, cited = [], []
    for label, classes in categories.items():
        elements = []
        for cls in classes:
            try:
                elements.extend(index.model.by_type(cls))
            except RuntimeError:
                pass
        elements = list({item.id(): item for item in elements}.values())
        if elements:
            rows.append(f"{label.title()}: {len(elements)} elements")
            cited.extend(elements)
    if not rows:
        return _not_found("No structural IFC element classes were found in the model.")
    return _text_answer(index, "Structural IFC elements by class (counts only; no load-path or efficiency inference):\n" + "\n".join(rows),
                        "counts of authored structural IFC entity classes", cited)


def _concrete_volume_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    if "concrete" not in q or "volume" not in q:
        return None
    classes = ("IfcBeam", "IfcColumn", "IfcSlab", "IfcWall", "IfcFooting", "IfcPile", "IfcMember", "IfcPlate")
    candidates = []
    for cls in classes:
        try:
            candidates.extend(index.model.by_type(cls))
        except RuntimeError:
            continue
    candidates = list({item.id(): item for item in candidates}.values())
    concrete_terms = ("concrete", "beton", "betong", "ortbeton", "stahlbeton", "reinforcedconcrete")
    selected = []
    for element in candidates:
        material_names = [name for name, _thickness in _materials(element)]
        type_name = _type_name(element)
        if any(term in _norm(name) for name in material_names for term in concrete_terms) or any(
                term in _norm(type_name) for term in concrete_terms):
            if "foundation" not in q or element.is_a() in {"IfcFooting", "IfcPile"} or (
                    element.is_a("IfcSlab") and str(getattr(element, "PredefinedType", "")).casefold() == "baseslab"):
                selected.append(element)
    if not selected:
        return _text_answer(index, "No concrete structural elements could be identified from authored material/type names.",
                            "search of structural IFC element materials and types for concrete terms", [])
    values = [(element, _prop_value(index, element, ("NetVolume", "GrossVolume", "Volume"), "VOLUMEUNIT"))
              for element in selected]
    known = [(element, value) for element, value in values if value is not None]
    if not known:
        return _not_found(f"Concrete elements were identified ({len(selected)}), but no authored volume quantities were found.")
    groups: Dict[str, List[float]] = {}
    for element, value in known:
        groups.setdefault(element.is_a(), []).append(value)
    rows = [f"{kind}: {sum(volumes):.2f} m3 across {len(volumes)} elements"
            for kind, volumes in sorted(groups.items())]
    total = sum(value for _, value in known)
    note = "" if len(known) == len(selected) else f"; partial because {len(selected) - len(known)} identified elements lack volume quantities"
    answer = f"Authored concrete volume: {total:.2f} m3{note}.\n" + "\n".join(rows)
    return _text_answer(index, answer, "sum of authored IFC volume quantities for elements with concrete material/type evidence",
                        [element for element, _ in known])


def _mep_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    terms = ("mep", "mechanical system", "heating system", "cooling system", "electrical component",
             "electrical load", "electrical distribution", "voltage level", "pipe", "piping", "duct", "lighting system",
             "sanitary equipment", "sanitary fixture", "sanitary furniture")
    if not any(term in q for term in terms):
        return None
    if "pipe" in q or "piping" in q or "duct" in q:
        category = "pipes" if "pipe" in q or "piping" in q else "ducts"
        elements = index.elements(category)
        if not elements:
            return _text_answer(index, f"No IfcPipeSegment/IfcPipeFitting or IfcDuctSegment/IfcDuctFitting elements are authored for {category}.",
                                f"IFC {category} entity search", [])
        if "length" in q:
            values = [(element, index.quantity(element, "Length", "LENGTHUNIT")[0]) for element in elements]
            known = [(element, value) for element, value in values if value is not None]
            if len(known) != len(elements):
                return _not_found(f"Length is authored for {len(known)} of {len(elements)} {category}; total withheld.")
            total = sum(value for _, value in known)
            return _text_answer(index, f"Total authored {category} length: {total:.2f} m across {len(elements)} elements.",
                                f"sum of IFC {category} Length quantities", elements)
        return _type_breakdown(index, category, question)

    try:
        systems = list(index.model.by_type("IfcDistributionSystem"))
    except RuntimeError:  # IFC2x3 calls these IfcSystem/IfcBuildingSystem
        systems = list(index.model.by_type("IfcSystem"))
    if systems:
        rows, cited = [], []
        for system in systems:
            members = [member for rel in (getattr(system, "IsGroupedBy", None) or [])
                       for member in (getattr(rel, "RelatedObjects", None) or [])]
            predefined = getattr(system, "PredefinedType", None)
            label = _named(system)
            if predefined and str(predefined) not in {"NOTDEFINED", "USERDEFINED"}:
                label += f" ({predefined})"
            rows.append(f"{label}: {len(members)} assigned components")
            cited.extend([system, *members])
        return _text_answer(index, "Authored IFC distribution systems:\n" + "\n".join(rows),
                            "IfcDistributionSystem group assignments", cited)
    elements = index.elements("mep components")
    if elements:
        groups: Dict[str, List[Any]] = {}
        for element in elements:
            groups.setdefault(element.is_a(), []).append(element)
        rows = [f"{name}: {len(items)} elements" for name, items in sorted(groups.items())]
        return _text_answer(index, "MEP component classes found (no system assignments authored):\n" + "\n".join(rows),
                            "counts of authored IFC flow component classes", elements)
    return _text_answer(index, "No IfcDistributionSystem or supported MEP component entities are authored in this model.",
                        "IFC distribution-system and MEP component entity search", [])


def _load_bearing_answer(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    if not any(term in q for term in ("load bearing", "loadbearing", "load-bearing", "main structural wall", "main structural walls")):
        return None
    walls = list(index.model.by_type("IfcWall"))
    if not walls:
        return _not_found("No IfcWall elements were found for load-bearing classification.")
    classified = []
    for wall in walls:
        value = _prop_raw(wall, ("LoadBearing", "IsLoadBearing", "LoadBearingFlag"))
        if value is not None:
            classified.append((wall, str(value).casefold() in {"true", "yes", "1"}))
    if not classified:
        return _text_answer(index, "Load-bearing classification is not authored for the walls in this IFC model.",
                            "search of IfcWall properties for LoadBearing/IsLoadBearing", walls)
    load_bearing = [wall for wall, value in classified if value]
    non_load_bearing = [wall for wall, value in classified if not value]
    unknown = len(walls) - len(classified)
    answer = (f"Load-bearing walls: {len(load_bearing)}; non-load-bearing walls: {len(non_load_bearing)}; "
              f"unclassified walls: {unknown}.")
    return _text_answer(index, answer, "authored IfcWall load-bearing properties", [wall for wall, _ in classified])


def _wall_quantity_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    wants_facade = any(term in q for term in ("facade", "façade", "exterior wall surface", "external wall surface"))
    wants_wall_area = any(term in q for term in ("wall surface", "wall area", "surface area of walls"))
    wants_length = "wall" in q and "length" in q
    if not (wants_facade or wants_wall_area or wants_length):
        return None
    walls = list({wall.id(): wall for wall in index.model.by_type("IfcWall")}.values())
    if not walls:
        return _not_found("No IfcWall elements were found for the requested wall quantity.")
    scope = None
    if any(term in q for term in ("interior", "inside", "internal")):
        scope = False
    elif wants_facade or any(term in q for term in ("exterior", "outside", "external")):
        scope = True
    if scope is not None:
        selected = [wall for wall in walls if _is_external(wall) is scope]
        if not selected:
            return _not_found("The model does not provide IsExternal classification for the requested wall scope.")
        walls = selected
    quantity_name = "Length" if wants_length else "NetSideArea"
    unit = "LENGTHUNIT" if wants_length else "AREAUNIT"
    values = [(wall, index.quantity(wall, quantity_name, unit)[0]) for wall in walls]
    known = [(wall, value) for wall, value in values if value is not None]
    if len(known) != len(walls):
        return _not_found(f"Only {len(known)} of {len(walls)} scoped walls have authored {quantity_name}; total withheld.")
    total = sum(value for _, value in known)
    scope_name = "exterior" if scope is True else "interior" if scope is False else "all"
    answer = f"Total {scope_name} wall {'length' if wants_length else 'net side area'}: {total:.2f} {'m' if wants_length else 'm2'} across {len(walls)} walls."
    return _text_answer(index, answer, f"sum of authored IfcWall {quantity_name} quantities", walls)


def _ceiling_material_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    if "ceiling" not in q or "material" not in q:
        return None
    candidates = []
    try:
        candidates.extend(index.model.by_type("IfcCovering"))
    except RuntimeError:
        pass
    candidates.extend(index.model.by_type("IfcSlab"))
    rows, cited = [], []
    for element in candidates:
        materials = _materials(element)
        if not materials:
            continue
        material_text = "; ".join(
            f"{name} ({thickness * 1000:.0f} mm)" if thickness is not None else name
            for name, thickness in materials
        )
        rows.append(f"{_type_name(element)} ({element.is_a()}): {material_text}")
        cited.append(element)
    if not rows:
        return _not_found("No material associations were authored on IfcCovering/IfcSlab ceiling candidates.")
    return _text_answer(index, "Materials on authored ceiling/slab candidates (IFC does not always classify ceiling adjacency):\n" + "\n".join(rows[:100]),
                        "material layer associations on IfcCovering and IfcSlab", cited[:100])


def _covering_detail(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    if not any(term in q for term in ("vapor barrier", "vapour barrier", "skirting", "skirting board")):
        return None
    coverings = index.elements("coverings")
    target_terms = ("vapor barrier", "vapour barrier", "vapourbarrier", "vaporbarrier", "membrane") if "barrier" in q else ("skirting", "skirtingboard", "baseboard")
    selected = [element for element in coverings
                if any(term.replace(" ", "") in _norm(" ".join(str(value or "") for value in (
                    _type_name(element), getattr(element, "Name", None), getattr(element, "PredefinedType", None)))).replace("_", "").replace(" ", "")
                       for term in target_terms)]
    if not selected:
        return _text_answer(index, f"No IfcCovering elements matching {('vapor barrier' if 'barrier' in q else 'skirting board')} are authored in the model.",
                            "IfcCovering type/name/predefined-type search", coverings)
    groups: Dict[str, List[Any]] = {}
    for item in selected:
        groups.setdefault(_type_name(item), []).append(item)
    rows = [f"{name}: {len(items)} elements" for name, items in sorted(groups.items())]
    if "length" in q:
        values = [(item, _prop_value(index, item, ("Length", "NetLength", "GrossLength"), "LENGTHUNIT")) for item in selected]
        known = [(item, value) for item, value in values if value is not None]
        if len(known) == len(selected):
            rows.append(f"Total authored length: {sum(value for _, value in known):.2f} m")
        else:
            rows.append(f"Length is authored for {len(known)} of {len(selected)} matching elements; total withheld.")
    return _text_answer(index, "Matching covering elements:\n" + "\n".join(rows),
                        "IfcCovering names, types and authored quantities", selected)


def _office_space_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    if "office" not in q and "buero" not in q:
        return None
    if not any(term in q for term in ("space", "room", "office")):
        return None
    spaces = [space for space in index.model.by_type("IfcSpace")
              if any(term in _norm(_named(space)) for term in ("office", "buero", "bureau", "kontor"))]
    if not spaces:
        return _not_found("No office-type IfcSpace names or room labels were found.")
    groups: Dict[str, List[Any]] = {}
    for space in spaces:
        groups.setdefault(_named(space), []).append(space)
    rows = [f"{name}: {len(items)} spaces" for name, items in sorted(groups.items())]
    return _text_answer(index, f"Office-type spaces identified by authored space names: {len(spaces)}\n" + "\n".join(rows),
                        "IfcSpace Name/LongName matching office terms", spaces)


def _bathroom_tiling_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    if not ("bathroom" in q and any(term in q for term in ("tile", "tiling", "flooring"))):
        return None
    spaces = [space for space in index.model.by_type("IfcSpace")
              if any(term in _norm(_named(space)) for term in ("bath", "wc", "shower"))]
    tile_coverings = []
    for covering in index.elements("coverings"):
        labels = _norm(" ".join(str(value or "") for value in (
            _type_name(covering), getattr(covering, "Name", None), getattr(covering, "PredefinedType", None))))
        materials = _norm(" ".join(name for name, _thickness in _materials(covering)))
        if "tile" in labels or "tile" in materials or "ceramic" in labels or "ceramic" in materials:
            tile_coverings.append(covering)
    if not spaces or not tile_coverings:
        return _not_found("Bathroom IfcSpace labels or tile/ceramic IfcCovering elements are not sufficiently authored to calculate tiled area.")
    rel_class = "IfcRelCoversSpaces"
    try:
        relations = index.model.by_type(rel_class)
    except RuntimeError:
        relations = []
    associations = []
    for relation in relations:
        space = getattr(relation, "RelatingSpace", None)
        if space not in spaces:
            continue
        for covering in getattr(relation, "RelatedCoverings", None) or []:
            if covering in tile_coverings:
                associations.append((space, covering))
    if not associations:
        return _not_found("Tile coverings were found, but the IFC model does not associate them with bathroom spaces.")
    by_room: Dict[int, Tuple[Any, List[Any]]] = {}
    for space, covering in associations:
        by_room.setdefault(space.id(), (space, []))[1].append(covering)
    rows, cited = [], []
    for space, coverings in by_room.values():
        areas = [_prop_value(index, covering, ("NetArea", "GrossArea", "Area"), "AREAUNIT") for covering in coverings]
        if any(area is None for area in areas):
            rows.append(f"{_named(space)}: {len(coverings)} tile coverings; area not authored")
        else:
            rows.append(f"{_named(space)}: {sum(area for area in areas if area is not None):.2f} m2")
        cited.extend([space, *coverings])
    return _text_answer(index, "Authored bathroom tile coverings by space:\n" + "\n".join(rows),
                        "IfcRelCoversSpaces associations and covering area quantities", cited)


def answer_ifc_text_question(question: str, index: IfcIndex) -> Dict[str, Any]:
    """Answer common IFC-Bench text questions from authored IFC semantics only."""
    q = _norm(question)
    metadata = _project_metadata(index, question)
    if metadata is not None:
        return metadata
    scale = _building_scale(index, question)
    if scale is not None:
        return scale
    window_area = _window_area_summary(index, question)
    if window_area is not None:
        return window_area
    if any(term in q for term in ("floor to floor", "floor-to-floor", "heights of each level", "elevations of the building levels", "elevation of the building levels", "level elevations")):
        return _level_heights(index) or _not_found("Storey elevations are insufficient to compute floor-to-floor heights.")
    ceiling = _ceiling_summary(index, question)
    if ceiling is not None:
        return ceiling
    ceiling_materials = _ceiling_material_summary(index, question)
    if ceiling_materials is not None:
        return ceiling_materials
    covering_detail = _covering_detail(index, question)
    if covering_detail is not None:
        return covering_detail
    bathroom_tiling = _bathroom_tiling_summary(index, question)
    if bathroom_tiling is not None:
        return bathroom_tiling
    office_spaces = _office_space_summary(index, question)
    if office_spaces is not None:
        return office_spaces
    load_bearing = _load_bearing_answer(index, question)
    if load_bearing is not None:
        return load_bearing
    if any(term in q for term in ("fire rating", "fire resistance", "glazing", "u value", "thermal transmittance",
                                   "capacity", "voltage", "energy efficiency", "maintenance", "warranty", "lifespan")):
        property_answer = _generic_property_answer(index, question)
        if property_answer is not None:
            return property_answer
    spatial = _space_summary(index, question)
    if spatial is not None:
        return spatial
    if "door" in q or "window" in q:
        category = "doors" if "door" in q else "windows"
        if any(term in q for term in ("width", "widest", "narrowest", "largest", "greatest", "average", "minimum", "maximum")):
            return _width_summary(index, category, question)
        if any(term in q for term in ("type", "model", "quantit", "how many")):
            return _type_breakdown(index, category)
    if "wall" in q or "insulation" in q:
        wall_answer = _wall_types(index, question)
        if wall_answer is not None:
            return wall_answer
    wall_quantity = _wall_quantity_summary(index, question)
    if wall_quantity is not None:
        return wall_quantity
    category_words = {
        "slabs": ("slab", "roof slab", "floor slab"),
        "columns": ("column", "pillar"),
        "beams": ("beam", "girder"),
        "stairs": ("stair", "staircase"),
        "sanitary fixtures": ("sanitary fixture", "sanitary equipment"),
        "coverings": ("covering", "coverings"),
        "railings": ("railing", "railings"),
        "furniture": ("furniture", "furnishing"),
        "members": ("structural member",),
        "plates": ("structural plate",),
        "curtain walls": ("curtain wall", "curtain walls"),
    }
    for category, terms in category_words.items():
        if any(term in q for term in terms):
            if any(term in q for term in ("type", "thickness", "dimension", "size", "quantity", "count", "how many", "which", "what")):
                return _type_breakdown(index, category, question)
    structural = _structural_summary(index, question)
    concrete = _concrete_volume_summary(index, question)
    if concrete is not None:
        return concrete
    if structural is not None:
        return structural
    mep = _mep_summary(index, question)
    if mep is not None:
        return mep
    generic = _generic_property_answer(index, question)
    if generic is not None:
        return generic
    return {"status": "unsupported", "reason": "No generic IFC text-query rule matched this question."}
