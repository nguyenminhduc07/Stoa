"""Extract structured geometry and text from DXF and DWG CAD files.

DXF is read directly with ezdxf. DWG is converted through ezdxf's ODA File
Converter adapter, which requires the ODA File Converter application installed
on the host. Entity handles, layers, layouts, bounds, and coordinates are kept
as provenance for downstream question answering.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _point(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        coords = list(value)
        result = [float(component) for component in coords[:3]]
        return result if all(math.isfinite(component) for component in result) else None
    except (TypeError, ValueError):
        return None


def _entity_bbox(entity: Any) -> Optional[Dict[str, List[float]]]:
    try:
        from ezdxf import bbox
        box = bbox.extents([entity], fast=True)
        if not box.has_data:
            return None
        return {"min": _point(box.extmin), "max": _point(box.extmax)}
    except Exception:
        # Proxy entities, malformed geometry, and unsupported entity types can
        # lack a computable extent; the entity record itself is still useful.
        return None


def _text_for_entity(entity: Any) -> Optional[str]:
    kind = entity.dxftype()
    try:
        if kind == "TEXT":
            return str(entity.dxf.text)
        if kind == "MTEXT":
            return entity.plain_text()
        if kind == "ATTRIB":
            return str(entity.dxf.text)
        if kind == "ATTDEF":
            return str(entity.dxf.prompt or entity.dxf.tag)
        if kind in {"DIMENSION", "LEADER", "MLEADER"}:
            text = getattr(entity.dxf, "text", "")
            return str(text) if text else None
    except (AttributeError, TypeError, ValueError):
        return None
    return None


def _entity_geometry(entity: Any) -> Dict[str, Any]:
    kind = entity.dxftype()
    dxf = entity.dxf
    try:
        if kind == "LINE":
            return {"start": _point(dxf.start), "end": _point(dxf.end)}
        if kind == "LWPOLYLINE":
            return {
                "vertices": [[float(v) for v in point] for point in entity.get_points("xyb")],
                "closed": bool(entity.closed),
            }
        if kind == "POLYLINE":
            return {
                "vertices": [_point(vertex.dxf.location) for vertex in entity.vertices],
                "closed": bool(entity.is_closed),
            }
        if kind in {"CIRCLE", "ARC"}:
            result = {"center": _point(dxf.center), "radius": float(dxf.radius)}
            if kind == "ARC":
                result.update({"start_angle": float(dxf.start_angle), "end_angle": float(dxf.end_angle)})
            return result
        if kind == "ELLIPSE":
            return {
                "center": _point(dxf.center), "major_axis": _point(dxf.major_axis),
                "ratio": float(dxf.ratio), "start_param": float(dxf.start_param),
                "end_param": float(dxf.end_param),
            }
        if kind in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}:
            return {
                "insert": _point(dxf.insert), "height": float(getattr(dxf, "height", 0.0) or 0.0),
                "rotation": float(getattr(dxf, "rotation", 0.0) or 0.0),
                "style": str(getattr(dxf, "style", "Standard")),
            }
        if kind == "INSERT":
            return {
                "insert": _point(dxf.insert), "block": str(dxf.name),
                "xscale": float(getattr(dxf, "xscale", 1.0)),
                "yscale": float(getattr(dxf, "yscale", 1.0)),
                "zscale": float(getattr(dxf, "zscale", 1.0)),
                "rotation": float(getattr(dxf, "rotation", 0.0)),
                "attributes": [
                    {"tag": str(att.dxf.tag), "text": str(att.dxf.text)}
                    for att in getattr(entity, "attribs", ())
                ],
            }
        if kind == "DIMENSION":
            try:
                measurement = float(entity.get_measurement())
            except (AttributeError, TypeError, ValueError):
                measurement = None
            return {
                "dimension_type": int(getattr(dxf, "dimtype", 0)),
                "measurement": measurement,
                "text_override": str(getattr(dxf, "text", "")),
                "definition_point": _point(getattr(dxf, "defpoint", None)),
            }
        if kind == "HATCH":
            return {
                "pattern_name": str(getattr(dxf, "pattern_name", "")),
                "path_count": len(entity.paths),
                "solid_fill": bool(getattr(dxf, "solid_fill", 0)),
            }
        if kind == "SPLINE":
            return {
                "degree": int(dxf.degree),
                "control_points": [_point(v) for v in entity.control_points],
                "fit_points": [_point(v) for v in entity.fit_points],
                "closed": bool(entity.closed),
            }
        if kind == "POINT":
            return {"location": _point(dxf.location)}
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return {}
    return {}


def _entity_record(entity: Any, layout_name: str) -> Dict[str, Any]:
    dxf = entity.dxf
    record = {
        "handle": str(getattr(dxf, "handle", "")),
        "type": entity.dxftype(),
        "layer": str(getattr(dxf, "layer", "0")),
        "layout": layout_name,
        "color_index": int(getattr(dxf, "color", 256) or 256),
        "linetype": str(getattr(dxf, "linetype", "ByLayer")),
        "lineweight": int(getattr(dxf, "lineweight", -1) or -1),
        "visible": not bool(getattr(dxf, "invisible", 0)),
        "text": _text_for_entity(entity),
        "geometry": _entity_geometry(entity),
        "bbox": _entity_bbox(entity),
    }
    if entity.dxftype() == "INSERT":
        record["block"] = str(dxf.name)
    return record


def _read_document(path: Path, use_oda_for_dwg: bool) -> Any:
    try:
        import ezdxf
    except ImportError as exc:
        raise RuntimeError("CAD parsing requires ezdxf. Install it with: pip install ezdxf") from exc

    if path.suffix.lower() == ".dxf":
        try:
            return ezdxf.readfile(path)
        except (OSError, ezdxf.DXFError) as exc:
            raise ValueError(f"Could not read DXF file {path}: {exc}") from exc
    if path.suffix.lower() == ".dwg":
        if not use_oda_for_dwg:
            raise RuntimeError("DWG conversion is disabled; pass use_oda_for_dwg=True to enable ODA conversion.")
        try:
            from ezdxf.addons import odafc
            return odafc.readfile(path)
        except ImportError as exc:
            raise RuntimeError("DWG conversion requires ezdxf's ODA File Converter adapter.") from exc
        except Exception as exc:
            raise RuntimeError(
                "Could not convert DWG. Install the ODA File Converter and make it discoverable to ezdxf; "
                "then retry. The DWG was not modified."
            ) from exc
    raise ValueError(f"Unsupported CAD extension: {path.suffix or '(none)'}")


def parse_cad(file_path: str | Path, *, use_oda_for_dwg: bool = True) -> Dict[str, Any]:
    """Extract modelspace/layout entities, layers, blocks, units, and bounds."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CAD file not found: {path}")
    document = _read_document(path, use_oda_for_dwg)
    import ezdxf

    entities: List[Dict[str, Any]] = []
    layout_counts: Dict[str, Counter] = {}
    for layout in document.layouts:
        name = str(layout.name)
        counter = Counter()
        for entity in layout:
            kind = entity.dxftype()
            counter[kind] += 1
            entities.append(_entity_record(entity, name))
        layout_counts[name] = counter

    layer_counts = Counter(record["layer"] for record in entities)
    layers = []
    for layer in document.layers:
        name = str(layer.dxf.name)
        layers.append({
            "name": name,
            "entity_count": layer_counts.get(name, 0),
            "color_index": int(getattr(layer.dxf, "color", 7)),
            "linetype": str(getattr(layer.dxf, "linetype", "CONTINUOUS")),
            "lineweight": int(getattr(layer.dxf, "lineweight", -1)),
            "is_off": bool(layer.is_off()),
            "is_frozen": bool(layer.is_frozen()),
            "is_locked": bool(getattr(layer.dxf, "locked", 0)),
        })

    blocks = []
    for block in document.blocks:
        name = str(block.name)
        if name.startswith("*"):
            continue
        counts = Counter(entity.dxftype() for entity in block)
        blocks.append({"name": name, "entity_count": sum(counts.values()), "entity_types": dict(counts)})

    try:
        unit_code = int(document.header.get("$INSUNITS", 0))
        unit_name = ezdxf.units.unit_name(unit_code)
    except (AttributeError, TypeError, ValueError):
        unit_code, unit_name = 0, "Unitless"

    bounds = None
    if entities:
        points = [record["bbox"] for record in entities if record.get("bbox")]
        if points:
            bounds = {
                "min": [min(item["min"][axis] for item in points) for axis in range(3)],
                "max": [max(item["max"][axis] for item in points) for axis in range(3)],
            }

    type_counts = Counter(record["type"] for record in entities)
    text_records = [record for record in entities if record.get("text")]
    return {
        "source": {"path": str(path), "format": path.suffix[1:].upper()},
        "dxf_version": str(document.dxfversion),
        "units": {"code": unit_code, "name": unit_name},
        "bounds": bounds,
        "entity_count": len(entities),
        "entity_types": dict(type_counts),
        "layouts": {name: dict(counts) for name, counts in layout_counts.items()},
        "layers": layers,
        "blocks": blocks,
        "text_entity_count": len(text_records),
        "entities": entities,
    }


def parse_dxf(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)
    if path.suffix.lower() != ".dxf":
        raise ValueError("parse_dxf expects a .dxf file")
    return parse_cad(path)


def parse_dwg(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)
    if path.suffix.lower() != ".dwg":
        raise ValueError("parse_dwg expects a .dwg file")
    return parse_cad(path, use_oda_for_dwg=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured entities from a DXF or DWG file")
    parser.add_argument("cad_file", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    data = parse_cad(args.cad_file)
    output = json.dumps(data, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)
