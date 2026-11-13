"""Extract a provenance-preserving index from an IFC building model.

Install IfcOpenShell with ``pip install ifcopenshell``. Parsing is deliberately
separate from question answering: call ``parse_ifc`` once per uploaded model,
cache the returned JSON-compatible structure for the conversation, then
retrieve/filter its objects for each question.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _value(value: Any) -> Any:
    """Convert IfcOpenShell values into JSON-compatible Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_value(item) for item in value]
    wrapped = getattr(value, "wrappedValue", None)
    if wrapped is not None:
        return _value(wrapped)
    if hasattr(value, "is_a"):
        return {"ifc_type": value.is_a(), "id": value.id()}
    return str(value)


def _unit_label(unit: Any) -> Optional[str]:
    if unit is None:
        return None
    name = getattr(unit, "Name", None)
    prefix = getattr(unit, "Prefix", None)
    if name:
        return f"{prefix or ''}{name}"
    return unit.is_a()


def _property_definitions(definition: Any) -> Iterable[Any]:
    """Yield property/quantity sets from IFC2x3 and IFC4 relationships."""
    if definition is None:
        return
    if definition.is_a("IfcPropertySet") or definition.is_a("IfcElementQuantity"):
        yield definition
        return
    # IFC4 can use IfcPropertySetDefinitionSet for one relationship target.
    for item in getattr(definition, "wrappedValue", ()) or ():
        if item.is_a("IfcPropertySet") or item.is_a("IfcElementQuantity"):
            yield item


def _read_sets(element: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
    properties: Dict[str, Any] = {}
    quantities: Dict[str, Any] = {}
    for rel in getattr(element, "IsDefinedBy", ()) or ():
        if not rel.is_a("IfcRelDefinesByProperties"):
            continue
        definition = getattr(rel, "RelatingPropertyDefinition", None)
        for pset in _property_definitions(definition):
            set_name = getattr(pset, "Name", None) or pset.is_a()
            entries: Dict[str, Any] = {}
            is_quantity_set = pset.is_a("IfcElementQuantity")
            for item in getattr(pset, "HasProperties", ()) or getattr(pset, "Quantities", ()) or ():
                item_name = getattr(item, "Name", None) or item.is_a()
                if is_quantity_set:
                    quantity_type = item.is_a()
                    value_attr = next(
                        (attr for attr in (
                            "LengthValue", "AreaValue", "VolumeValue", "CountValue",
                            "WeightValue", "TimeValue",
                        ) if getattr(item, attr, None) is not None),
                        None,
                    )
                    entries[item_name] = {
                        "value": _value(getattr(item, value_attr)) if value_attr else None,
                        "quantity_type": quantity_type,
                        "unit": _unit_label(getattr(item, "Unit", None)),
                        "description": _value(getattr(item, "Description", None)),
                    }
                elif item.is_a("IfcPropertySingleValue"):
                    entries[item_name] = {
                        "value": _value(getattr(item, "NominalValue", None)),
                        "unit": _unit_label(getattr(item, "Unit", None)),
                        "property_type": item.is_a(),
                    }
                elif item.is_a("IfcPropertyEnumeratedValue"):
                    entries[item_name] = {
                        "value": _value(getattr(item, "EnumerationValues", None)),
                        "property_type": item.is_a(),
                    }
                else:
                    entries[item_name] = {"property_type": item.is_a()}
            target = quantities if is_quantity_set else properties
            target[set_name] = entries
    return properties, quantities


def _storey_for(element: Any) -> Optional[Dict[str, Any]]:
    for rel in getattr(element, "ContainedInStructure", ()) or ():
        structure = getattr(rel, "RelatingStructure", None)
        if structure and structure.is_a("IfcBuildingStorey"):
            return {"name": _value(getattr(structure, "Name", None)),
                    "guid": _value(getattr(structure, "GlobalId", None))}
    # Some models nest spaces/elements under spatial containers by aggregation.
    current = element
    visited = set()
    while current is not None and current.id() not in visited:
        visited.add(current.id())
        parent = None
        for rel in getattr(current, "Decomposes", ()) or ():
            parent = getattr(rel, "RelatingObject", None)
            if parent:
                break
        if parent and parent.is_a("IfcBuildingStorey"):
            return {"name": _value(getattr(parent, "Name", None)),
                    "guid": _value(getattr(parent, "GlobalId", None))}
        current = parent
    return None


def parse_ifc(file_path: str | Path) -> Dict[str, Any]:
    """Parse an IFC file into project metadata and searchable object records.

    Each record retains its model path and IFC GUID so an answer can cite the
    exact element and property/quantity set that supplied its evidence.
    Quantity values are preserved as authored in the IFC, along with explicit
    units when present. No geometry-derived quantities are invented here.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"IFC model not found: {path}")
    if path.suffix.lower() != ".ifc":
        raise ValueError(f"This parser supports IFC files, not {path.suffix or 'extensionless files'}")
    try:
        import ifcopenshell
    except ImportError as exc:
        raise RuntimeError("IFC parsing requires IfcOpenShell. Install it with: pip install ifcopenshell") from exc

    model = ifcopenshell.open(str(path))
    try:
        from ifcopenshell.util.unit import calculate_unit_scale
        length_scale = calculate_unit_scale(model)
    except (ImportError, AttributeError, RuntimeError, ValueError):
        length_scale = None

    project = next(iter(model.by_type("IfcProject")), None)
    storeys = []
    for storey in model.by_type("IfcBuildingStorey"):
        storeys.append({
            "name": _value(getattr(storey, "Name", None)),
            "guid": _value(getattr(storey, "GlobalId", None)),
            "elevation": _value(getattr(storey, "Elevation", None)),
            "description": _value(getattr(storey, "Description", None)),
        })

    objects: List[Dict[str, Any]] = []
    for element in model.by_type("IfcProduct"):
        guid = getattr(element, "GlobalId", None)
        if not guid:
            continue
        properties, quantities = _read_sets(element)
        objects.append({
            "guid": guid,
            "ifc_type": element.is_a(),
            "name": _value(getattr(element, "Name", None)),
            "long_name": _value(getattr(element, "LongName", None)),
            "description": _value(getattr(element, "Description", None)),
            "object_type": _value(getattr(element, "ObjectType", None)),
            "storey": _storey_for(element),
            "properties": properties,
            "quantities": quantities,
            "source": {"model": str(path), "ifc_id": element.id(), "guid": guid},
        })

    units = []
    if model.by_type("IfcUnitAssignment"):
        for unit in model.by_type("IfcUnitAssignment")[0].Units:
            units.append({"type": unit.is_a(), "name": _unit_label(unit)})

    return {
        "source": {"model": str(path), "format": "IFC"},
        "schema": getattr(model, "schema", None),
        "project": {
            "name": _value(getattr(project, "Name", None)) if project else None,
            "description": _value(getattr(project, "Description", None)) if project else None,
            "properties": _read_sets(project)[0] if project else {},
        },
        "units": {"assigned": units, "length_to_metres": length_scale},
        "storeys": storeys,
        "objects": objects,
        "object_count": len(objects),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract searchable data from an IFC model")
    parser.add_argument("ifc_file", help="Path to an .ifc model")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    result = parse_ifc(args.ifc_file)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    else:
        print(serialized)
