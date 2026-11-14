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


def _space_area(index: IfcIndex, element: Any, *, gross: bool = False) -> Optional[float]:
    names = ("GrossFloorArea", "GrossArea", "GrossFootprintArea") if gross else ("NetFloorArea", "NetArea", "NetFootprintArea", "GSA BIM Area")
    for name in names:
        value, _method = index.quantity(element, name, "AREAUNIT")
        if value is not None:
            return value
    return _prop_value(index, element, names, "AREAUNIT")


def _width(index: IfcIndex, element: Any) -> Optional[float]:
    value = _prop_value(index, element, ("OverallWidth", "ClearWidth", "NominalWidth", "Width"), "LENGTHUNIT")
    if value is not None:
        return value
    value, _method = index.quantity(element, "Width", "LENGTHUNIT")
    return value


def _text_evidence(index: IfcIndex, elements: Sequence[Any], **extra: Any) -> List[Dict[str, Any]]:
    return [_evidence(index, elements, **extra)] if elements else [{"source": index.path, **extra}]


def _text_answer(index: IfcIndex, answer: str, method: str, elements: Sequence[Any] = (), **extra: Any) -> Dict[str, Any]:
    return {"status": "answered", "answer": answer, "method": method,
            "evidence": _text_evidence(index, elements, **extra)}


def _find_question_storey(question: str, storeys: Sequence[Any]) -> Optional[Any]:
    q = _norm(question)
    matches = []
    for storey in storeys:
        name = str(getattr(storey, "Name", "") or "")
        if name and _norm(name) in q:
            matches.append(storey)
    return max(matches, key=lambda item: len(str(item.Name)), default=None)


def _storey_spaces(index: IfcIndex, storey: Any) -> List[Any]:
    return [space for space in index.model.by_type("IfcSpace") if index.storey_of(space) == storey.Name]


def _space_breakdown(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    model = index.model
    storeys = list(model.by_type("IfcBuildingStorey"))
    spaces = list(model.by_type("IfcSpace"))
    if not spaces:
        return _not_found("The IFC model contains no IfcSpace elements to list.")
    requested = _find_question_storey(question, storeys)
    selected_storeys = [requested] if requested else sorted(
        storeys, key=lambda s: (getattr(s, "Elevation", None) is None, getattr(s, "Elevation", 0) or 0)
    )
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
    for lower, upper in zip(levels, levels[1:]):
        elevation_a, elevation_b = float(lower.Elevation) * scale, float(upper.Elevation) * scale
        rows.append(f"{_named(lower)} to {_named(upper)}: {elevation_b - elevation_a:.2f} m")
    answer = "Floor-to-floor heights from stored storey elevations:\n" + "\n".join(rows)
    return _text_answer(index, answer, "IfcBuildingStorey.Elevation differences", levels)


def _space_summary(index: IfcIndex, question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    spaces = list(index.model.by_type("IfcSpace"))
    if not spaces:
        return _not_found("No IfcSpace elements are defined in this model.")
    if "largest room" in q or "largest space" in q or "biggest room" in q:
        values = [(area, space) for space in spaces if (area := _space_area(index, space)) is not None]
        if not values:
            return _not_found("No authored room area was found, so the largest space cannot be identified.")
        area, space = max(values, key=lambda pair: pair[0])
        return _text_answer(index, f"Largest space: {_named(space)}, {area:.2f} m2.",
                            "largest IfcSpace by authored net floor area", [space])
    if any(term in q for term in ("space breakdown", "space breakdown", "all spaces", "room breakdown", "spaces on", "rooms on", "space distribution", "spaces by level", "space by level")):
        return _space_breakdown(index, question)
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


def _type_breakdown(index: IfcIndex, category: str) -> Optional[Dict[str, Any]]:
    elements = index.elements(category)
    if not elements:
        return _not_found(f"No {category} are defined in this IFC model.")
    groups: Dict[str, List[Any]] = {}
    for element in elements:
        groups.setdefault(_type_name(element), []).append(element)
    lines = [f"Total {category}: {len(elements)}"]
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
        lines.append(f"- {name}: {len(members)}{suffix}")
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
        limit = int(limit_match.group(1)) if limit_match else 5
        selected = sorted(widths, key=lambda pair: pair[0], reverse=True)[:limit]
        cited = [e for _, e in selected]
        rows = [f"{i}. {_named(element)}: {width:.3f} m" for i, (width, element) in enumerate(selected, 1)]
        return _text_answer(index, "Largest authored widths:\n" + "\n".join(rows),
                            f"top {len(selected)} {category} widths from IFC properties", cited)
    else:
        value = max(width for width, _ in widths)
        selected = [e for width, e in widths if abs(width - value) < 1e-6]
        cited, label = selected, "Maximum"
    return _text_answer(index, f"{label} authored width: {value:.3f} m.",
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
    if not any(term in q for term in ("wall type", "wall material", "wall composition", "material composition", "insulation", "thickness of the wall")):
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
            exterior = [item for item in members
                        if str(_prop_value(index, item, ("IsExternal",))).casefold() == "true"]
            members = exterior
        elif "interior" in q or "partition" in q:
            interior = [item for item in members
                        if str(_prop_value(index, item, ("IsExternal",))).casefold() == "false"]
            members = interior
        if not members:
            continue
        layers = _materials(members[0])
        layer_text = "; ".join(f"{name} ({thickness:.3f} m)" if thickness is not None else name for name, thickness in layers)
        rows.append(f"- {type_name}: {len(members)} walls" + (f"; layers: {layer_text}" if layer_text else ""))
        cited.extend(members)
    return _text_answer(index, "Wall types and available material layers:\n" + "\n".join(rows),
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

    if any(term in q for term in ("project information", "project name", "building name", "location", "coordinates", "building type", "building function", "building usage")):
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
        if values:
            return _text_answer(index, "\n".join(values), "IfcProject/IfcBuilding/IfcSite authored attributes", cited)
        return _not_found("Project, building, and site metadata are not authored in this IFC model.")
    return None


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
    return _not_found(f"No matching authored properties ({', '.join(requested)}) were found.")


def answer_ifc_text_question(question: str, index: IfcIndex) -> Dict[str, Any]:
    """Answer common IFC-Bench text questions from authored IFC semantics only."""
    q = _norm(question)
    metadata = _project_metadata(index, question)
    if metadata is not None:
        return metadata
    if any(term in q for term in ("floor to floor", "floor-to-floor", "heights of each level")):
        return _level_heights(index) or _not_found("Storey elevations are insufficient to compute floor-to-floor heights.")
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
    category_words = {
        "slabs": ("slab", "roof slab", "floor slab"),
        "columns": ("column", "pillar"),
        "beams": ("beam", "girder"),
        "stairs": ("stair", "staircase"),
        "sanitary fixtures": ("sanitary fixture", "sanitary equipment"),
    }
    for category, terms in category_words.items():
        if any(term in q for term in terms) and any(term in q for term in ("type", "thickness", "dimension", "size")):
            return _type_breakdown(index, category)
    generic = _generic_property_answer(index, question)
    if generic is not None:
        return generic
    return {"status": "unsupported", "reason": "No generic IFC text-query rule matched this question."}
