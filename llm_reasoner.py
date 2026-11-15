"""Grounded LLM natural-language processing and compositional reasoning over drawing evidence.

Instead of matching questions against hardcoded keyword lists or templates, local
parsers extract structured, provenance-tagged EvidencePacks (every room, symbol,
space, CAD polygon, BIM element cohort, schedule row, and OCR text line gets a
stable ID mapped in Python to its exact bbox / GUID / entity handle and exact
numeric quantities). The LLM reasons over that inventory to select the matching
evidence IDs and aggregation operation, and Python deterministically computes
the final numeric result and attaches ground-truth citations.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DOTENV_CACHE: Optional[Dict[str, str]] = None
_DISCOVERED_GEMINI_MODELS: Optional[List[str]] = None
_WORKING_GEMINI_MODEL: Optional[str] = None


def _load_dotenv() -> Dict[str, str]:
    global _DOTENV_CACHE
    if _DOTENV_CACHE is not None:
        return _DOTENV_CACHE
    loaded: Dict[str, str] = {}
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        try:
            for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                if key:
                    loaded[key] = value
        except OSError:
            pass
    _DOTENV_CACHE = loaded
    return loaded


def get_env_setting(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    if val is not None and val.strip():
        return val.strip()
    return _load_dotenv().get(name, default).strip()


def gemini_api_key() -> Optional[str]:
    key = get_env_setting("GEMINI_API_KEY") or get_env_setting("GOOGLE_API_KEY")
    if key:
        return key
    stoa_key = get_env_setting("STOA_LLM_API_KEY")
    if stoa_key and not stoa_key.startswith("sk-"):
        return stoa_key
    return None


def openai_api_key() -> Optional[str]:
    stoa_key = get_env_setting("STOA_LLM_API_KEY")
    if stoa_key.startswith("sk-"):
        return stoa_key
    key = get_env_setting("OPENAI_API_KEY")
    return key or None


def is_configured() -> bool:
    return bool(gemini_api_key() or openai_api_key())


def active_provider_name() -> str:
    if gemini_api_key():
        return "gemini"
    if openai_api_key():
        return "openai"
    return "none"


@dataclass
class EvidenceNode:
    node_id: str
    kind: str
    label: str
    count: int = 1
    attributes: Dict[str, Optional[float]] = field(default_factory=dict)
    incomplete_attributes: set[str] = field(default_factory=set)
    method: str = "parser_evidence"
    review_recommended: bool = False
    citations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvidencePack:
    file_id: str
    file_name: str
    format: str
    drawing_type: Optional[str]
    catalog: Dict[str, Any]
    nodes: Dict[str, EvidenceNode]


def _box_contains_point(box: Sequence[float], x: float, y: float) -> bool:
    if len(box) != 4:
        return False
    return float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3])


def _scale_bbox(bbox: Optional[Sequence[float]], sx: float, sy: float) -> Optional[List[float]]:
    if not bbox or len(bbox) != 4:
        return None
    if sx == 1.0 and sy == 1.0:
        return [round(float(v), 2) if isinstance(v, float) else int(v) for v in bbox]
    return [
        round(float(bbox[0]) * sx, 2),
        round(float(bbox[1]) * sy, 2),
        round(float(bbox[2]) * sx, 2),
        round(float(bbox[3]) * sy, 2),
    ]


# --------------------------------------------------------------------------- Evidence Builders

def build_image_evidence_pack(
    file_id: str,
    file_name: str,
    document: Dict[str, Any],
    *,
    source_override: Optional[str] = None,
    page: int = 1,
    page_size: Optional[List[float]] = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    id_prefix: str = "",
) -> EvidencePack:
    """Build a structured EvidencePack from a parsed raster drawing (or rendered PDF page)."""
    source = source_override or document.get("source", {}).get("path") or file_name
    drawing_type = document.get("drawing_type") or "uncertain"
    width = int(document.get("width") or 0)
    height = int(document.get("height") or 0)
    total_px = max(1, width * height)
    coord_sys = (
        "PDF points; origin at top-left"
        if page_size is not None
        else "image pixels; origin at top-left"
    )
    unconfirmed_type = drawing_type != "floor_plan"

    nodes: Dict[str, EvidenceNode] = {}
    ocr_lines = document.get("lines") or []

    def make_citation(bbox: Optional[Sequence[float]], extra: Dict[str, Any]) -> Dict[str, Any]:
        cit: Dict[str, Any] = {
            "source": source,
            "page": page,
            "bbox": _scale_bbox(bbox, scale_x, scale_y),
            "coordinate_system": coord_sys,
        }
        if page_size is not None:
            cit["page_size"] = page_size
        cit.update({k: v for k, v in extra.items() if v is not None})
        return cit

    specialist = document.get("floorplan_specialist") or {}
    has_specialist = bool(specialist)
    model_name = specialist.get("model", "cubicasa5k_segmentation")

    indoor_rooms_cat: List[Dict[str, Any]] = []
    room_summary: Dict[str, Dict[str, Any]] = {}
    for idx, room in enumerate(specialist.get("rooms") or [], start=1):
        nid = f"{id_prefix}R{idx}"
        r_class = str(room.get("class") or "Room")
        bbox = room.get("bbox")
        area_px = int(room.get("area_px") or 0)
        conf = float(room.get("confidence") or 0.0)
        inside_texts: List[str] = []
        if bbox and len(bbox) == 4:
            for line in ocr_lines:
                lb = line.get("bbox")
                if lb and len(lb) == 4:
                    cx, cy = (lb[0] + lb[2]) / 2.0, (lb[1] + lb[3]) / 2.0
                    if _box_contains_point(bbox, cx, cy) and line.get("text"):
                        inside_texts.append(str(line["text"]))
        entry = {
            "id": nid,
            "room_type": r_class,
            "indoor": True,
            "area_pct_of_image": round(100.0 * area_px / total_px, 2),
            "confidence": round(conf, 3),
        }
        if inside_texts:
            entry["ocr_texts_inside"] = inside_texts[:6]
        indoor_rooms_cat.append(entry)
        bucket = room_summary.setdefault(r_class, {"count": 0, "ids": []})
        bucket["count"] += 1
        bucket["ids"].append(nid)
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="segmented_indoor_room",
            label=r_class,
            count=1,
            method="cubicasa5k_segmentation",
            review_recommended=True,
            citations=[make_citation(bbox, {"class": r_class, "confidence": round(conf, 4)})],
        )

    outdoor_cat: List[Dict[str, Any]] = []
    for idx, space in enumerate(specialist.get("outdoor_spaces") or [], start=1):
        nid = f"{id_prefix}O{idx}"
        bbox = space.get("bbox")
        area_px = int(space.get("area_px") or 0)
        conf = float(space.get("confidence") or 0.0)
        outdoor_cat.append({
            "id": nid,
            "space_type": "Outdoor",
            "indoor": False,
            "area_pct_of_image": round(100.0 * area_px / total_px, 2),
            "confidence": round(conf, 3),
        })
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="segmented_outdoor_space",
            label="Outdoor",
            count=1,
            method="cubicasa5k_segmentation",
            review_recommended=True,
            citations=[make_citation(bbox, {"class": "Outdoor", "confidence": round(conf, 4)})],
        )

    symbols_cat: List[Dict[str, Any]] = []
    symbol_summary: Dict[str, Dict[str, Any]] = {}
    for idx, icon in enumerate(specialist.get("icons") or [], start=1):
        nid = f"{id_prefix}I{idx}"
        s_class = str(icon.get("class") or "Symbol")
        bbox = icon.get("bbox")
        conf = float(icon.get("confidence") or 0.0)
        symbols_cat.append({
            "id": nid,
            "symbol_type": s_class,
            "confidence": round(conf, 3),
        })
        bucket = symbol_summary.setdefault(s_class, {"count": 0, "ids": []})
        bucket["count"] += 1
        bucket["ids"].append(nid)
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="segmented_symbol",
            label=s_class,
            count=1,
            method="cubicasa5k_segmentation",
            review_recommended=True,
            citations=[make_citation(bbox, {"class": s_class, "confidence": round(conf, 4)})],
        )

    ocr_labels_cat: List[Dict[str, Any]] = []
    ocr_label_summary: Dict[str, Dict[str, Any]] = {}
    for idx, hit in enumerate(document.get("ocr_room_labels") or [], start=1):
        nid = f"{id_prefix}L{idx}"
        cat = str(hit.get("category") or "room_label")
        lbl = str(hit.get("label") or "")
        txt = str(hit.get("ocr_text") or lbl)
        conf = float(hit.get("confidence") or 0.0)
        bbox = hit.get("bbox")
        ocr_labels_cat.append({
            "id": nid,
            "category": cat,
            "label": lbl,
            "ocr_text": txt,
            "confidence": round(conf, 3),
        })
        bucket = ocr_label_summary.setdefault(cat, {"count": 0, "ids": []})
        bucket["count"] += 1
        bucket["ids"].append(nid)
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="ocr_room_label",
            label=f"{cat}:{lbl}",
            count=1,
            method="room_label_ocr",
            review_recommended=unconfirmed_type or conf < 0.75,
            citations=[make_citation(bbox, {"label": lbl, "ocr_text": txt, "confidence": round(conf, 3)})],
        )

    text_lines_cat: List[Dict[str, Any]] = []
    measurements_set = set(str(m) for m in (document.get("measurements") or []))
    for idx, line in enumerate(ocr_lines[:100], start=1):
        txt = str(line.get("text") or "").strip()
        if not txt:
            continue
        nid = f"{id_prefix}T{idx}"
        bbox = line.get("bbox")
        conf = float(line.get("confidence") or 0.0)
        is_dim = any(m in txt for m in measurements_set) or bool(
            re.search(r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|m|m2|m²)\b", txt, re.I)
        )
        text_lines_cat.append({
            "id": nid,
            "text": txt,
            "is_dimension": is_dim,
        })
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="ocr_text_line",
            label=txt,
            count=1,
            method="raster_ocr_text",
            review_recommended=conf < 0.7,
            citations=[make_citation(bbox, {"ocr_text": txt, "ocr_confidence": round(conf, 3)})],
        )

    catalog: Dict[str, Any] = {
        "file_id": file_id,
        "file_name": file_name,
        "format": "pdf_rendered_page" if page_size is not None else "image",
        "page": page,
        "drawing_type": drawing_type,
        "image_dimensions_px": [width, height],
        "floorplan_segmentation_available": has_specialist,
        "segmentation_model": model_name if has_specialist else None,
        "segmentation_capabilities": {
            "segmented_room_classes": [
                "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", "Storage", "Garage", "Other rooms"
            ] if has_specialist else [],
            "segmented_symbol_classes": [
                "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
                "Sink", "Sauna bench", "Fireplace", "Bathtub", "Chimney",
            ] if has_specialist else [],
            "limitations": [
                "CubiCasa5K segments Bathtubs, Toilets, and Sinks, but has NO Shower icon class (abstain with not_found for standalone shower counts on raster images unless explicitly labeled in text).",
                "Raster images do not have calibrated metric scale unless printed dimensions explicitly give a measurement; do not fabricate room areas in m2 from pixel areas.",
                "NEVER use ocr_room_labels (L*) or ocr_text_lines (T*) to derive total/general room counts, bedroom/kitchen/bathroom counts, or areas — room counts MUST be derived strictly from geometric room regions (R*). Use L* ONLY for label-specific room categories not segmented as separate classes (wc_rooms, saunas, balconies). If floorplan_segmentation_available is false and the user asks for general rooms or geometric room/symbol counts, return action='not_found'.",
            ],
        },
        "total_segmented_indoor_rooms": len(indoor_rooms_cat),
        "indoor_room_summary_by_type": room_summary,
        "segmented_indoor_rooms": indoor_rooms_cat,
        "segmented_outdoor_spaces": outdoor_cat,
        "symbol_summary_by_type": symbol_summary,
        "segmented_symbols": symbols_cat,
        "ocr_label_summary_by_category": ocr_label_summary,
        "ocr_room_labels": ocr_labels_cat,
        "title_block_metadata": document.get("drawing_metadata") or {},
        "ocr_text_lines": (
            [t for t in text_lines_cat if t.get("is_dimension")][:20]
            + [t for t in text_lines_cat if not t.get("is_dimension")][:25]
        ),
    }
    if document.get("floorplan_specialist_error"):
        catalog["floorplan_specialist_error"] = document["floorplan_specialist_error"]

    return EvidencePack(
        file_id=file_id,
        file_name=file_name,
        format="image",
        drawing_type=drawing_type,
        catalog=catalog,
        nodes=nodes,
    )


def build_cad_evidence_pack(
    file_id: str,
    file_name: str,
    fmt: str,
    floor_index: Dict[str, Any],
    generic_cad: Optional[Dict[str, Any]] = None,
) -> EvidencePack:
    """Build a structured EvidencePack from DXF/DWG semantic floor index + generic CAD entities."""
    from cad_qa import OUTDOOR_WORDS, SHAFT_VOID_WORDS, _inside

    source = floor_index.get("source", {}).get("path") or file_name
    units_label = floor_index.get("source", {}).get("units") or "m"
    nodes: Dict[str, EvidenceNode] = {}

    units_cat: List[Dict[str, Any]] = []
    raw_units = floor_index.get("units") or []
    for idx, unit in enumerate(raw_units, start=1):
        nid = f"U{idx}"
        u_lbl = unit.get("label") or f"Unit_{idx}"
        units_cat.append({
            "id": nid,
            "apartment_unit": u_lbl,
            "handle": unit.get("handle"),
        })
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="cad_apartment_unit",
            label=f"Apartment {u_lbl}",
            count=1,
            method="dxf_unit_outline",
            citations=[{
                "source": source,
                "handle": unit.get("handle"),
                "layer": unit.get("layer", "UNIT_OUTLINE"),
                "bbox": unit.get("bbox"),
                "unit": u_lbl,
            }],
        )

    def _unit_for_bbox(bbox: Optional[Sequence[float]]) -> Optional[str]:
        if not bbox or len(bbox) != 4:
            return None
        cx, cy = (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0
        for u in raw_units:
            if u.get("label") and u.get("polygon") and _inside((cx, cy), u["polygon"]):
                return str(u["label"])
        return None

    rooms_cat: List[Dict[str, Any]] = []
    for idx, room in enumerate(floor_index.get("rooms") or [], start=1):
        nid = f"R{idx}"
        r_name = room.get("name")
        r_num = room.get("number")
        r_unit = room.get("unit")
        r_area = float(room.get("area") or 0.0)
        name_cf = (r_name or "").casefold()
        is_outdoor = any(w in name_cf for w in OUTDOOR_WORDS)
        is_shaft = any(w in name_cf for w in SHAFT_VOID_WORDS)
        is_indoor_counted = bool((r_name or r_num) and not is_outdoor and not is_shaft)
        rooms_cat.append({
            "id": nid,
            "room_number": r_num,
            "room_name": r_name,
            "apartment_unit": r_unit,
            "area_m2": round(r_area, 4),
            "is_outdoor": is_outdoor,
            "is_shaft_or_void": is_shaft,
            "is_standard_indoor_room": is_indoor_counted,
        })
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="cad_room",
            label=f"{r_name or 'Room'} ({r_num or nid})",
            count=1,
            attributes={"area_m2": r_area, "net_floor_area_m2": r_area},
            method="dxf_room_polygon",
            citations=[{
                "source": source,
                "handle": room.get("handle"),
                "label_handles": room.get("label_handles"),
                "layer": room.get("layer", "ROOM_OUTLINE"),
                "bbox": room.get("bbox"),
                "room_name": r_name,
                "room_number": r_num,
                "unit": r_unit,
                "area_m2": round(r_area, 4),
            }],
        )

    doors_cat: List[Dict[str, Any]] = []
    for idx, door in enumerate(floor_index.get("doors") or [], start=1):
        nid = f"D{idx}"
        d_unit = _unit_for_bbox(door.get("bbox"))
        doors_cat.append({"id": nid, "apartment_unit": d_unit})
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="cad_door",
            label="Door",
            count=1,
            method="dxf_door_layer",
            citations=[{
                "source": source,
                "handle": door.get("handle"),
                "layer": door.get("layer", "DOOR"),
                "bbox": door.get("bbox"),
            }],
        )

    windows_cat: List[Dict[str, Any]] = []
    for idx, win in enumerate(floor_index.get("windows") or [], start=1):
        nid = f"W{idx}"
        w_unit = _unit_for_bbox(win.get("bbox"))
        windows_cat.append({"id": nid, "apartment_unit": w_unit})
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="cad_window",
            label="Window",
            count=1,
            method="dxf_window_layer",
            citations=[{
                "source": source,
                "handle": win.get("handle"),
                "layer": win.get("layer", "WINDOW"),
                "bbox": win.get("bbox"),
            }],
        )

    fixtures_cat: List[Dict[str, Any]] = []
    fixture_summary: Dict[str, Dict[str, Any]] = {}
    for idx, fix in enumerate(floor_index.get("fixtures") or [], start=1):
        nid = f"F{idx}"
        f_kind = str(fix.get("kind") or "fixture")
        f_unit = _unit_for_bbox(fix.get("bbox"))
        fixtures_cat.append({"id": nid, "fixture_type": f_kind, "apartment_unit": f_unit})
        bucket = fixture_summary.setdefault(f_kind, {"count": 0, "ids": []})
        bucket["count"] += 1
        bucket["ids"].append(nid)
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="cad_fixture",
            label=f_kind,
            count=1,
            method="dxf_fixture_geometry",
            citations=[{
                "source": source,
                "handle": fix.get("handle"),
                "layer": fix.get("layer", "FIXTURE"),
                "bbox": fix.get("bbox"),
                "fixture_type": f_kind,
            }],
        )

    layers_cat: List[Dict[str, Any]] = []
    texts_cat: List[Dict[str, Any]] = []
    dims_cat: List[Dict[str, Any]] = []
    if generic_cad:
        for idx, layer in enumerate(generic_cad.get("layers") or [], start=1):
            cnt = int(layer.get("entity_count") or 0)
            if cnt <= 0:
                continue
            nid = f"LY{idx}"
            l_name = str(layer.get("name") or "")
            layers_cat.append({"id": nid, "layer_name": l_name, "entity_count": cnt})
            nodes[nid] = EvidenceNode(
                node_id=nid,
                kind="cad_layer",
                label=l_name,
                count=1,
                method="cad_layer_inventory",
                citations=[{"source": source, "layer": l_name, "entity_count": cnt}],
            )
        t_idx = 0
        d_idx = 0
        for ent in generic_cad.get("entities") or []:
            etype = ent.get("type")
            if etype == "DIMENSION" and d_idx < 60:
                meas = (ent.get("geometry") or {}).get("measurement")
                txt = ent.get("text")
                if meas is not None or txt:
                    d_idx += 1
                    nid = f"DM{d_idx}"
                    dims_cat.append({
                        "id": nid,
                        "layer": ent.get("layer"),
                        "text": txt,
                        "measurement": round(float(meas), 4) if isinstance(meas, (int, float)) else None,
                    })
                    nodes[nid] = EvidenceNode(
                        node_id=nid,
                        kind="cad_dimension",
                        label=str(txt or meas),
                        count=1,
                        attributes={"length_m": float(meas)} if isinstance(meas, (int, float)) else {},
                        method="cad_dimension_entity",
                        citations=[{
                            "source": source,
                            "handle": ent.get("handle"),
                            "layer": ent.get("layer"),
                            "bbox": ent.get("bbox"),
                            "measurement": meas,
                        }],
                    )
            elif ent.get("text") and t_idx < 80:
                t_idx += 1
                nid = f"TX{t_idx}"
                txt_val = str(ent["text"]).strip()
                texts_cat.append({"id": nid, "layer": ent.get("layer"), "text": txt_val})
                nodes[nid] = EvidenceNode(
                    node_id=nid,
                    kind="cad_text",
                    label=txt_val,
                    count=1,
                    method="cad_text_entity",
                    citations=[{
                        "source": source,
                        "handle": ent.get("handle"),
                        "layer": ent.get("layer"),
                        "bbox": ent.get("bbox"),
                        "text": txt_val,
                    }],
                )

    catalog = {
        "file_id": file_id,
        "file_name": file_name,
        "format": fmt,
        "units": units_label,
        "conventions": [
            "For indoor room counts or apartment floor areas on CAD plans, select rooms where is_standard_indoor_room is true (excludes outdoor balconies/terraces/loggias and shafts/voids and unlabeled polygons) unless outdoor spaces or shafts are explicitly requested.",
            "Sanitary fixtures include toilet, washbasin, bathtub, and shower.",
        ],
        "summary": {
            "apartment_units_count": len(units_cat),
            "standard_indoor_rooms_count": sum(1 for r in rooms_cat if r["is_standard_indoor_room"]),
            "all_room_polygons_count": len(rooms_cat),
            "doors_count": len(doors_cat),
            "windows_count": len(windows_cat),
            "all_door_ids": [d["id"] for d in doors_cat],
            "all_window_ids": [w["id"] for w in windows_cat],
            "fixtures_by_type": fixture_summary,
        },
        "apartment_units": units_cat,
        "rooms": rooms_cat,
        "doors": doors_cat,
        "windows": windows_cat,
        "sanitary_fixtures": fixtures_cat,
        "cad_layers": layers_cat,
        "cad_dimensions": dims_cat[:40],
        "cad_texts": texts_cat[:40],
    }
    return EvidencePack(
        file_id=file_id,
        file_name=file_name,
        format=fmt,
        drawing_type="floor_plan",
        catalog=catalog,
        nodes=nodes,
    )


def build_ifc_evidence_pack(
    file_id: str,
    file_name: str,
    parsed_ifc: Dict[str, Any],
    ifc_index: Any,
) -> EvidencePack:
    """Build a structured EvidencePack from an IFC model (spaces + disjoint element cohorts)."""
    import ifcopenshell.util.element as E

    source = parsed_ifc.get("source", {}).get("path") or file_name
    nodes: Dict[str, EvidenceNode] = {}

    storeys_cat: List[Dict[str, Any]] = []
    for idx, st in enumerate(parsed_ifc.get("storeys") or [], start=1):
        nid = f"ST{idx}"
        st_name = str(st.get("name") or f"Storey_{idx}")
        storeys_cat.append({
            "id": nid,
            "name": st_name,
            "elevation": st.get("elevation"),
            "guid": st.get("guid"),
        })
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="ifc_storey",
            label=st_name,
            count=1,
            method="authored IfcBuildingStorey",
            citations=[{"source": source, "guid": st.get("guid"), "storey": st_name}],
        )

    spaces_cat: List[Dict[str, Any]] = []
    s_idx = 0
    for obj in parsed_ifc.get("objects") or []:
        if obj.get("ifc_type") != "IfcSpace":
            continue
        s_idx += 1
        nid = f"S{s_idx}"
        storey_name = (obj.get("storey") or {}).get("name")
        area_val: Optional[float] = None
        area_method = "missing"
        for set_name, qset in (obj.get("quantities") or {}).items():
            for qname, q in (qset or {}).items():
                if "NetFloorArea" in qname and isinstance(q.get("value"), (int, float)):
                    area_val = float(q["value"])
                    area_method = f"authored {set_name}.{qname}"
                    break
            if area_val is not None:
                break
        if area_val is None:
            comp = (obj.get("computed_quantities") or {}).get("NetFloorArea")
            if comp and isinstance(comp.get("value"), (int, float)):
                area_val = float(comp["value"])
                area_method = str(comp.get("method") or "geometry footprint fallback")

        props = obj.get("properties") or {}
        cat_val = None
        for pset in props.values():
            if isinstance(pset, dict) and "Category" in pset and isinstance(pset["Category"], dict):
                cat_val = pset["Category"].get("value")
                if cat_val:
                    break

        attrs: Dict[str, Optional[float]] = {}
        incomplete: set[str] = set()
        if area_val is not None:
            attrs["net_floor_area_m2"] = area_val
            attrs["area_m2"] = area_val
        else:
            incomplete.update({"net_floor_area_m2", "area_m2"})

        s_label = str(obj.get("label") or obj.get("name") or nid)
        spaces_cat.append({
            "id": nid,
            "storey": storey_name,
            "name": obj.get("name"),
            "long_name": obj.get("long_name"),
            "label": s_label,
            "category": cat_val,
            "net_floor_area_m2": round(area_val, 4) if area_val is not None else None,
            "area_source": area_method,
        })
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="ifc_space",
            label=s_label,
            count=1,
            attributes=attrs,
            incomplete_attributes=incomplete,
            method=area_method,
            citations=[{
                "source": source,
                "guid": obj.get("guid"),
                "ifc_id": (obj.get("source") or {}).get("ifc_id"),
                "ifc_type": "IfcSpace",
                "name": s_label,
                "storey": storey_name,
            }],
        )

    categories_order = (
        "walls", "doors", "windows", "slabs", "stairs", "columns", "beams",
        "sanitary fixtures", "coverings", "railings", "furniture", "members",
        "plates", "curtain walls", "pipes", "ducts",
    )
    seen_elem_ids: set[int] = set()
    cohorts_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def _bool_prop(psets: Dict[str, Dict[str, Any]], prop_name: str) -> Optional[bool]:
        for pset in psets.values():
            if prop_name in pset and isinstance(pset[prop_name], bool):
                return pset[prop_name]
        return None

    def _str_prop(psets: Dict[str, Dict[str, Any]], prop_name: str) -> Optional[str]:
        for pset in psets.values():
            val = pset.get(prop_name)
            if val is not None and str(val).strip():
                return str(val).strip()
        return None

    def _material_name(elem: Any) -> Optional[str]:
        try:
            mat = E.get_material(elem, should_skip_usage=True)
            if mat is None:
                return None
            if hasattr(mat, "Name") and mat.Name:
                return str(mat.Name)
            for attr in ("Materials", "MaterialLayers", "MaterialProfiles", "MaterialConstituents"):
                items = getattr(mat, attr, None)
                if items:
                    names = []
                    for it in items:
                        m = getattr(it, "Material", it)
                        if hasattr(m, "Name") and m.Name:
                            names.append(str(m.Name))
                    if names:
                        return " + ".join(dict.fromkeys(names))
        except Exception:
            pass
        return None

    def _ingest_element(cat_label: str, elem: Any) -> None:
        eid = elem.id()
        if eid in seen_elem_ids:
            return
        seen_elem_ids.add(eid)
        ifc_type = elem.is_a()
        storey = ifc_index.storey_of(elem)
        psets = E.get_psets(elem, psets_only=True)
        is_ext = _bool_prop(psets, "IsExternal")
        load_bearing = _bool_prop(psets, "LoadBearing")
        fire_rating = _str_prop(psets, "FireRating")
        et = E.get_type(elem)
        type_name = str(getattr(et, "Name", None) or getattr(elem, "ObjectType", None) or "")[:80] or None
        mat_name = _material_name(elem)

        key = (cat_label, ifc_type, storey, is_ext, load_bearing, type_name, mat_name, fire_rating)
        bucket = cohorts_map.get(key)
        if bucket is None:
            bucket = {
                "category": cat_label,
                "ifc_type": ifc_type,
                "storey": storey,
                "is_external": is_ext,
                "load_bearing": load_bearing,
                "type_name": type_name,
                "material": mat_name,
                "fire_rating": fire_rating,
                "elements": [],
            }
            cohorts_map[key] = bucket
        bucket["elements"].append(elem)

    for cat in categories_order:
        try:
            elems = ifc_index.elements(cat)
        except Exception:
            elems = []
        for el in elems:
            _ingest_element(cat, el)

    try:
        for el in ifc_index.elements("mep components"):
            _ingest_element("other_mep_components", el)
    except Exception:
        pass

    cohorts_cat: List[Dict[str, Any]] = []
    for g_idx, bucket in enumerate(cohorts_map.values(), start=1):
        nid = f"G{g_idx}"
        elems = bucket["elements"]
        cat_label = bucket["category"]
        cnt = len(elems)

        attrs: Dict[str, Optional[float]] = {}
        incomplete: set[str] = set()
        methods_used: set[str] = set()

        if cat_label == "walls":
            lengths = [ifc_index.quantity(e, "Length", "LENGTHUNIT") for e in elems]
            areas = [ifc_index.quantity(e, "NetSideArea", "AREAUNIT") for e in elems]
            vols = [ifc_index.quantity(e, "NetVolume", "VOLUMEUNIT") for e in elems]
            if all(v is not None for v, _ in lengths):
                attrs["length_m"] = sum(float(v) for v, _ in lengths if v is not None)
                methods_used.update(m for _, m in lengths if m)
            else:
                incomplete.add("length_m")
            if all(v is not None for v, _ in areas):
                attrs["side_area_m2"] = sum(float(v) for v, _ in areas if v is not None)
                attrs["area_m2"] = attrs["side_area_m2"]
                methods_used.update(m for _, m in areas if m)
            else:
                incomplete.update({"side_area_m2", "area_m2"})
            if all(v is not None for v, _ in vols):
                attrs["volume_m3"] = sum(float(v) for v, _ in vols if v is not None)
                methods_used.update(m for _, m in vols if m)
            else:
                incomplete.add("volume_m3")
        elif cat_label == "slabs":
            vols = [ifc_index.quantity(e, "NetVolume", "VOLUMEUNIT") for e in elems]
            if all(v is not None for v, _ in vols):
                attrs["volume_m3"] = sum(float(v) for v, _ in vols if v is not None)
                methods_used.update(m for _, m in vols if m)
            else:
                incomplete.add("volume_m3")

        cohort_entry: Dict[str, Any] = {
            "id": nid,
            "category": cat_label,
            "ifc_type": bucket["ifc_type"],
            "storey": bucket["storey"],
            "count": cnt,
        }
        for opt_key in ("is_external", "load_bearing", "type_name", "material", "fire_rating"):
            if bucket[opt_key] is not None:
                cohort_entry[opt_key] = bucket[opt_key]
        if "length_m" in attrs and attrs["length_m"] is not None:
            cohort_entry["total_length_m"] = round(float(attrs["length_m"]), 4)
        if "side_area_m2" in attrs and attrs["side_area_m2"] is not None:
            cohort_entry["total_side_area_m2"] = round(float(attrs["side_area_m2"]), 4)
        if "volume_m3" in attrs and attrs["volume_m3"] is not None:
            cohort_entry["total_volume_m3"] = round(float(attrs["volume_m3"]), 4)
        if incomplete:
            cohort_entry["missing_quantities"] = sorted(incomplete)
        cohorts_cat.append(cohort_entry)

        cits = [
            {
                "source": source,
                "guid": getattr(e, "GlobalId", None),
                "ifc_id": e.id(),
                "ifc_type": e.is_a(),
                "name": getattr(e, "Name", None),
                "storey": bucket["storey"],
            }
            for e in elems[:60]
        ]
        method_label = "; ".join(sorted(methods_used)) if methods_used else f"ifc_{cat_label.replace(' ', '_')}"
        nodes[nid] = EvidenceNode(
            node_id=nid,
            kind="ifc_element_cohort",
            label=f"{cnt} {bucket['ifc_type']} ({cat_label})",
            count=cnt,
            attributes=attrs,
            incomplete_attributes=incomplete,
            method=method_label,
            citations=cits,
        )

    category_totals: Dict[str, int] = {}
    for c in cohorts_cat:
        category_totals[c["category"]] = category_totals.get(c["category"], 0) + int(c["count"])

    spaces_by_storey: Dict[str, Dict[str, Any]] = {}
    for sp in spaces_cat:
        st_key = str(sp.get("storey") or "Unassigned")
        st_bucket = spaces_by_storey.setdefault(st_key, {"count": 0, "total_area_m2": 0.0, "space_ids": []})
        st_bucket["count"] += 1
        if isinstance(sp.get("net_floor_area_m2"), (int, float)):
            st_bucket["total_area_m2"] = round(st_bucket["total_area_m2"] + float(sp["net_floor_area_m2"]), 4)
        st_bucket["space_ids"].append(sp["id"])

    catalog = {
        "file_id": file_id,
        "file_name": file_name,
        "format": "ifc",
        "schema": parsed_ifc.get("schema"),
        "project": parsed_ifc.get("project"),
        "units": parsed_ifc.get("units"),
        "conventions": [
            "Spaces (S*) represent rooms/zones (IfcSpace) with net_floor_area_m2.",
            "Element cohorts (G*) are disjoint groups of building elements; count_selected sums each selected cohort's count.",
            "For MEP total counts, sum sanitary fixtures + pipes + ducts + other_mep_components.",
            "If a requested quantity is listed in missing_quantities on any required cohort/space, return action='not_found' rather than returning an incomplete partial sum.",
        ],
        "storeys": storeys_cat,
        "spaces_count": len(spaces_cat),
        "spaces_by_storey_summary": spaces_by_storey,
        "element_totals_by_category": category_totals,
        "spaces": spaces_cat[:160],
        "element_cohorts": cohorts_cat,
    }
    return EvidencePack(
        file_id=file_id,
        file_name=file_name,
        format="ifc",
        drawing_type="bim_model",
        catalog=catalog,
        nodes=nodes,
    )


def build_pdf_evidence_pack(
    file_id: str,
    file_name: str,
    pdf_doc: Dict[str, Any],
    rendered_pages: Optional[Dict[int, Dict[str, Any]]] = None,
) -> EvidencePack:
    """Build a structured EvidencePack from PDF tables, text, dimensions, and rendered visual sheets."""
    from pdf_parser import (
        _door_type_description,
        _expand_merged_schedule_rows,
        _is_border_dump_table,
        _table_header,
        _table_rows,
    )

    source = pdf_doc.get("source", {}).get("path") or file_name
    pages = pdf_doc.get("pages") or []
    rendered_pages = rendered_pages or {}
    nodes: Dict[str, EvidenceNode] = {}

    all_tables = [table for p in pages for table in (p.get("tables") or []) if not _is_border_dump_table(table)]
    header_candidates = [t for t in all_tables if len(_table_rows(t)) <= 4]

    schedule_rows_cat: List[Dict[str, Any]] = []
    sr_idx = 0
    for page_obj in pages:
        p_num = int(page_obj.get("page") or 1)
        p_size = [float(page_obj.get("width") or 0), float(page_obj.get("height") or 0)]
        for t_idx, table in enumerate(page_obj.get("tables") or [], start=1):
            if _is_border_dump_table(table):
                continue
            raw_rows = _table_rows(table)
            if len(raw_rows) < 2:
                continue
            headers = _table_header(table, header_candidates)
            if not any(headers):
                continue
            t_bbox = table.get("bbox")
            raw_row_objs = table.get("rows") or []
            for r_i, texts in _expand_merged_schedule_rows(raw_rows):
                cells_raw = raw_row_objs[r_i].get("cells") or [] if r_i < len(raw_row_objs) else []
                sr_idx += 1
                nid = f"SR{sr_idx}"
                mapped: Dict[str, str] = {}
                row_bboxes = [c.get("bbox") for c in cells_raw if c.get("bbox")]
                row_bbox = (
                    [
                        min(b[0] for b in row_bboxes),
                        min(b[1] for b in row_bboxes),
                        max(b[2] for b in row_bboxes),
                        max(b[3] for b in row_bboxes),
                    ]
                    if row_bboxes
                    else t_bbox
                )
                for c_idx, val in enumerate(texts):
                    if not val:
                        continue
                    col_name = headers[c_idx] if c_idx < len(headers) and headers[c_idx] else f"col_{c_idx + 1}"
                    mapped[col_name] = val
                schedule_rows_cat.append({
                    "id": nid,
                    "page": p_num,
                    "table_index": t_idx,
                    "cells": mapped,
                })
                nodes[nid] = EvidenceNode(
                    node_id=nid,
                    kind="pdf_schedule_row",
                    label=json.dumps(mapped, ensure_ascii=False)[:140],
                    count=1,
                    method="pdf_schedule_table",
                    citations=[{
                        "source": source,
                        "page": p_num,
                        "page_size": p_size,
                        "bbox": row_bbox,
                        "table_bbox": t_bbox,
                        "coordinate_system": "PDF points; origin at top-left",
                        "row": mapped,
                    }],
                )

    door_legends_cat: List[Dict[str, Any]] = []
    seen_codes: set[str] = set()
    for page_obj in pages:
        for w in page_obj.get("words") or []:
            txt = str(w.get("text") or "").strip().upper()
            if re.fullmatch(r"[A-Z]\d", txt) and txt not in seen_codes:
                seen_codes.add(txt)
                desc = _door_type_description(txt, pages)
                if desc:
                    nid = f"DL{len(door_legends_cat) + 1}"
                    dp = int(desc.get("page") or 1)
                    p_match = next((p for p in pages if int(p.get("page") or 1) == dp), page_obj)
                    p_size = [float(p_match.get("width") or 0), float(p_match.get("height") or 0)]
                    door_legends_cat.append({
                        "id": nid,
                        "code": txt,
                        "page": dp,
                        "description": desc["text"],
                    })
                    nodes[nid] = EvidenceNode(
                        node_id=nid,
                        kind="pdf_door_legend",
                        label=f"{txt}: {desc['text']}",
                        count=1,
                        method="pdf_door_elevation_legend",
                        citations=[{
                            "source": source,
                            "page": dp,
                            "page_size": p_size,
                            "bbox": desc.get("bbox"),
                            "coordinate_system": "PDF points; origin at top-left",
                            "text": desc["text"],
                        }],
                    )

    pages_overview: List[Dict[str, Any]] = []
    page_texts_cat: List[Dict[str, Any]] = []
    pt_idx = 0
    for page_obj in pages:
        p_num = int(page_obj.get("page") or 1)
        p_w = float(page_obj.get("width") or 0)
        p_h = float(page_obj.get("height") or 0)
        words = page_obj.get("words") or []
        vec_count = (
            int(page_obj.get("vector_line_count") or 0)
            + int(page_obj.get("vector_rect_count") or 0)
            + int(page_obj.get("vector_curve_count") or 0)
        )
        pages_overview.append({
            "page": p_num,
            "size_pt": [round(p_w, 1), round(p_h, 1)],
            "word_count": len(words),
            "table_count": len(page_obj.get("tables") or []),
            "vector_object_count": vec_count,
            "image_count": int(page_obj.get("image_count") or 0),
            "visual_analysis_ready": p_num in rendered_pages,
        })
        if words and pt_idx < 120:
            from image_parser import _ocr_lines
            synthetic = [
                {
                    "text": w["text"],
                    "bbox": [float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"])],
                    "confidence": float(w.get("confidence", 1.0)),
                }
                for w in words
                if all(k in w for k in ("text", "x0", "top", "x1", "bottom"))
            ]
            for line in _ocr_lines(synthetic)[:60]:
                if pt_idx >= 120:
                    break
                txt = str(line.get("text") or "").strip()
                if not txt:
                    continue
                pt_idx += 1
                nid = f"PT{pt_idx}"
                page_texts_cat.append({"id": nid, "page": p_num, "text": txt})
                nodes[nid] = EvidenceNode(
                    node_id=nid,
                    kind="pdf_text_line",
                    label=txt,
                    count=1,
                    method="pdf_embedded_text",
                    citations=[{
                        "source": source,
                        "page": p_num,
                        "page_size": [p_w, p_h],
                        "bbox": line.get("bbox"),
                        "coordinate_system": "PDF points; origin at top-left",
                        "text": txt,
                    }],
                )

    visual_pages_cat: Dict[str, Any] = {}
    for p_num, rendered_info in sorted(rendered_pages.items()):
        img_doc = rendered_info["document"]
        p_w, p_h = rendered_info["page_width"], rendered_info["page_height"]
        sx, sy = rendered_info["scale_x"], rendered_info["scale_y"]
        sub_pack = build_image_evidence_pack(
            file_id=file_id,
            file_name=file_name,
            document=img_doc,
            source_override=source,
            page=p_num,
            page_size=[p_w, p_h],
            scale_x=sx,
            scale_y=sy,
            id_prefix=f"P{p_num}_",
        )
        nodes.update(sub_pack.nodes)
        visual_pages_cat[f"page_{p_num}"] = {
            "drawing_type": sub_pack.catalog.get("drawing_type"),
            "floorplan_segmentation_available": sub_pack.catalog.get("floorplan_segmentation_available"),
            "total_segmented_indoor_rooms": sub_pack.catalog.get("total_segmented_indoor_rooms"),
            "indoor_room_summary_by_type": sub_pack.catalog.get("indoor_room_summary_by_type"),
            "segmented_indoor_rooms": sub_pack.catalog.get("segmented_indoor_rooms"),
            "segmented_outdoor_spaces": sub_pack.catalog.get("segmented_outdoor_spaces"),
            "symbol_summary_by_type": sub_pack.catalog.get("symbol_summary_by_type"),
            "segmented_symbols": sub_pack.catalog.get("segmented_symbols"),
            "ocr_label_summary_by_category": sub_pack.catalog.get("ocr_label_summary_by_category"),
            "ocr_room_labels": sub_pack.catalog.get("ocr_room_labels"),
        }

    catalog = {
        "file_id": file_id,
        "file_name": file_name,
        "format": "pdf",
        "page_count": len(pages),
        "pages_overview": pages_overview,
        "conventions": [
            "Use schedule_rows (SR*), door_legends (DL*), and page_text_lines (PT*) for schedule, specification, title-block, and printed-dimension questions.",
            "If the question asks about spatial floor-plan geometry, room counts, or graphical symbol counts on a PDF page where visual_analysis_ready is false (and no IFC/DXF twin answers it), return action='render_pdf_page' with file_id and pdf_page_to_render.",
            "If visual_floorplan_pages is populated for the target page, reason over its P{page}_R*, P{page}_I*, and P{page}_L* items just like an image floor plan.",
        ],
        "schedule_rows_count": len(schedule_rows_cat),
        "schedule_rows": schedule_rows_cat[:120],
        "door_legends": door_legends_cat,
        "page_text_lines": page_texts_cat[:60],
        "visual_floorplan_pages": visual_pages_cat,
    }
    return EvidencePack(
        file_id=file_id,
        file_name=file_name,
        format="pdf",
        drawing_type=None,
        catalog=catalog,
        nodes=nodes,
    )


# --------------------------------------------------------------------------- LLM Grounded Reasoning

_REASONING_SYSTEM_PROMPT = """You are the grounded reasoning engine for Stoa, an architectural & BIM construction drawing QA assistant.
You receive:
1. The user's natural-language question (in any language: English, Finnish, German, French, Estonian, Danish, etc.).
2. Recent conversation history (for follow-up context).
3. Structured EvidencePacks extracted by specialist local parsers (IFC, CAD DXF/DWG, PDF schedules/text/renders, and Image Floor-Plan segmentation + OCR).

CORE RULES (STRICT):
1. ZERO HALLUCINATION: Never invent numbers, room names, dimensions, or IDs. Every fact MUST come from the provided EvidencePacks, and `selected_ids` MUST contain the exact `id` strings from the chosen file's EvidencePack.
2. PYTHON COMPUTES ARITHMETIC:
   - For counting objects/rooms/rows/cohorts (`operation="count_selected"`): select ALL matching IDs in `selected_ids`. Python will sum their counts deterministically!
     * Example: When asked "how many rooms are there?" on an image floor plan, select ALL indoor room IDs (`R1`, `R2`, ...) across every detected indoor room type (`Bedroom`, `Living Room`, `Kitchen`, `Bath`, `Hallway`, `Storage`, `Garage`, `Other rooms`, `Room`), excluding `Outdoor` spaces (`O*`) unless outdoor spaces were asked. NEVER count `ocr_room_labels` (`L*`) or `ocr_text_lines` (`T*`) to answer general/total room counts — room counts MUST come strictly from geometric room regions (`R*`).
     * Example: When asked "how many rooms excluding hallways and storage?", select all `R*` IDs except `Hallway` and `Storage`.
     * Example: Use `ocr_room_labels` (`L*`) ONLY when asked specifically for WC rooms, saunas, or balconies on a raster plan. If `segmented_indoor_rooms` (`R*`) is empty/unavailable and the user asks for total rooms, bedrooms, kitchens, bathrooms, doors, or windows, return `action="not_found"` — NEVER estimate a room count from text labels!
   - For summing areas, lengths, or volumes (`operation="sum_attribute"`): set `attribute` to `"area_m2"`, `"net_floor_area_m2"`, `"length_m"`, `"side_area_m2"`, or `"volume_m3"`, and list ALL matching IDs in `selected_ids`. Python will sum the exact unrounded floats!
     * If ANY required item/cohort has that attribute missing/null (`missing_quantities`), return `action="not_found"` instead of a partial sum.
   - For largest/smallest (`operation="max_attribute"` or `"min_attribute"`): select the candidate pool in `selected_ids` and set `attribute`. If the user asks for the numeric value, set `unit` (e.g. `"m2"`); if the user asks *which* room/element is largest/smallest, set `direct_answer` to that specific winner's name + value and put ONLY the winning item's ID in `selected_ids` with `operation="direct_value"`.
   - For text lookups, schedule cell queries, comparisons, breakdowns, or single property lookups (`operation="direct_value"`): write the exact concise answer in `direct_answer`, and put the supporting evidence IDs in `selected_ids`.
3. FORMAT PRIORITY FOR TWINS:
   - If multiple files represent the same drawing/building (`related_files`), prefer the highest-precision format that contains the needed data: `ifc > dxf > dwg > pdf > image`.
   - If multiple UNRELATED drawings could answer the question and neither the question nor conversation history specifies which file, return `action="clarification_required"` and ask a brief `clarification_question`.
4. ON-DEMAND PDF VISUAL ANALYSIS:
   - If the best matching file is a `pdf` and the question asks about floor-plan rooms, symbols, or visual layout on a page where `visual_analysis_ready` is `false` (and no IFC/DXF twin answers it), return `action="render_pdf_page"` with `file_id` and `pdf_page_to_render` (1-based page number; if multiple drawing pages exist and page is ambiguous, return `clarification_required`).
5. ABSTAIN RATHER THAN GUESS:
   - If a condition/filter cannot be verified from the EvidencePack (e.g., color "red doors", acoustic rating not in properties, standalone "shower" count on CubiCasa5K raster without text labels, or uncalibrated m2 area on a raster image), return `action="not_found"` or `action="unsupported"` with a clear `explanation`.
6. CLEAR EXPLANATION:
   - Always provide a concise, helpful `explanation` (1-2 sentences, in the user's language) summarizing how the answer was derived (e.g., "Summed all 8 detected indoor rooms: 3 Bedroom, 1 Living Room, 1 Kitchen, 1 Bath, 1 Hallway, and 1 Storage (1 Outdoor space excluded).").
"""

_REASONING_GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": ["answer", "not_found", "clarification_required", "unsupported", "render_pdf_page"],
        },
        "file_id": {"type": "STRING", "nullable": True},
        "pdf_page_to_render": {"type": "INTEGER", "nullable": True},
        "operation": {
            "type": "STRING",
            "enum": ["count_selected", "sum_attribute", "min_attribute", "max_attribute", "direct_value"],
        },
        "selected_ids": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "attribute": {
            "type": "STRING",
            "enum": ["none", "area_m2", "net_floor_area_m2", "length_m", "side_area_m2", "volume_m3"],
        },
        "unit": {"type": "STRING", "nullable": True},
        "direct_answer": {"type": "STRING", "nullable": True},
        "explanation": {"type": "STRING"},
        "clarification_question": {"type": "STRING", "nullable": True},
    },
    "required": [
        "action",
        "file_id",
        "pdf_page_to_render",
        "operation",
        "selected_ids",
        "attribute",
        "unit",
        "direct_answer",
        "explanation",
        "clarification_question",
    ],
}

_REASONING_OPENAI_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["answer", "not_found", "clarification_required", "unsupported", "render_pdf_page"],
        },
        "file_id": {"type": ["string", "null"]},
        "pdf_page_to_render": {"type": ["integer", "null"]},
        "operation": {
            "type": "string",
            "enum": ["count_selected", "sum_attribute", "min_attribute", "max_attribute", "direct_value"],
        },
        "selected_ids": {"type": "array", "items": {"type": "string"}},
        "attribute": {
            "type": "string",
            "enum": ["none", "area_m2", "net_floor_area_m2", "length_m", "side_area_m2", "volume_m3"],
        },
        "unit": {"type": ["string", "null"]},
        "direct_answer": {"type": ["string", "null"]},
        "explanation": {"type": "string"},
        "clarification_question": {"type": ["string", "null"]},
    },
    "required": [
        "action",
        "file_id",
        "pdf_page_to_render",
        "operation",
        "selected_ids",
        "attribute",
        "unit",
        "direct_answer",
        "explanation",
        "clarification_question",
    ],
    "additionalProperties": False,
}


def _discover_gemini_models(api_key: str) -> List[str]:
    """Query Google ModelService.ListModels for active generateContent Gemini models."""
    global _DISCOVERED_GEMINI_MODELS
    if _DISCOVERED_GEMINI_MODELS is not None:
        return _DISCOVERED_GEMINI_MODELS
    url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=100"
    req = urllib.request.Request(url, headers={"x-goog-api-key": api_key}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    usable: List[Tuple[Tuple[int, int, float, str], str]] = []
    for item in payload.get("models") or []:
        methods = item.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        raw_name = str(item.get("name") or "")
        name = raw_name[len("models/"):] if raw_name.startswith("models/") else raw_name
        low = name.casefold()
        if "gemini" not in low:
            continue
        if any(skip in low for skip in (
            "embedding", "tts", "image", "audio", "robotics", "computer-use", "aqa", "thinking", "deep-research"
        )):
            continue
        # Prefer fast flash / flash-lite models first, newest major.minor version first, stable before preview/exp.
        tier = (
            0 if "flash" in low and "lite" not in low else
            1 if "flash" in low else
            2 if "pro" in low else
            3
        )
        stable = 0 if "exp" not in low and "preview" not in low else 1
        m_ver = re.search(r"gemini-(\d+(?:\.\d+)?)", low)
        ver = float(m_ver.group(1)) if m_ver else 0.0
        usable.append(((tier, stable, -ver, name), name))

    usable.sort(key=lambda pair: pair[0])
    discovered = [name for _, name in usable]
    if discovered:
        _DISCOVERED_GEMINI_MODELS = discovered
    return discovered


def _call_gemini_reasoner(api_key: str, payload_json: str) -> Dict[str, Any]:
    configured = (get_env_setting("GEMINI_MODEL") or get_env_setting("STOA_LLM_MODEL") or _DEFAULT_GEMINI_MODEL).strip()
    if configured.startswith("models/"):
        configured = configured[len("models/"):]
    if configured and not configured.startswith("gemini"):
        configured = _DEFAULT_GEMINI_MODEL

    models_to_try = list(dict.fromkeys([
        *([_WORKING_GEMINI_MODEL] if _WORKING_GEMINI_MODEL else []),
        configured or _DEFAULT_GEMINI_MODEL,
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ]))

    def _build_body(model_name: str, *, disable_thinking: bool = True) -> bytes:
        low_model = model_name.casefold()
        gen_cfg: Dict[str, Any] = {
            "temperature": 0,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "responseSchema": _REASONING_GEMINI_SCHEMA,
        }
        if disable_thinking:
            if "gemini-3" in low_model or "-3." in low_model or "-3-" in low_model:
                gen_cfg["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}
            elif "2.5-pro" in low_model:
                gen_cfg["thinkingConfig"] = {"thinkingBudget": 128}
            elif "2.5" in low_model:
                gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
        return json.dumps({
            "systemInstruction": {"parts": [{"text": _REASONING_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": payload_json}]}],
            "generationConfig": gen_cfg,
        }).encode("utf-8")

    try:
        timeout_sec = max(10.0, float(get_env_setting("STOA_LLM_TIMEOUT_SECONDS") or "45"))
    except ValueError:
        timeout_sec = 45.0

    data: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    tried: set[str] = set()

    def _try_models(candidates: List[str]) -> Optional[Dict[str, Any]]:
        global _WORKING_GEMINI_MODEL
        nonlocal last_error
        for model in candidates:
            if model in tried:
                continue
            tried.add(model)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model, safe='')}:generateContent"
            for disable_thinking in (True, False):
                req = urllib.request.Request(
                    url,
                    data=_build_body(model, disable_thinking=disable_thinking),
                    headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                        parsed = json.loads(resp.read().decode("utf-8"))
                    _WORKING_GEMINI_MODEL = model
                    return parsed
                except urllib.error.HTTPError as exc:
                    err_body = ""
                    try:
                        err_body = exc.read().decode("utf-8", errors="ignore").strip()
                    except Exception:
                        pass
                    detail = err_body[:400] if err_body else exc.reason
                    if exc.code == 400 and disable_thinking and "thinking" in detail.casefold():
                        continue
                    last_error = f"HTTP {exc.code} for model '{model}': {detail}"
                    if _WORKING_GEMINI_MODEL == model:
                        _WORKING_GEMINI_MODEL = None
                    if exc.code in {404, 408, 429, 500, 502, 503, 504}:
                        break
                    raise RuntimeError(last_error) from exc
                except (TimeoutError, urllib.error.URLError, OSError) as exc:
                    last_error = f"{type(exc).__name__} for model '{model}': {exc}"
                    if _WORKING_GEMINI_MODEL == model:
                        _WORKING_GEMINI_MODEL = None
                    break
        return None

    data = _try_models(models_to_try)
    if data is None:
        discovered = _discover_gemini_models(api_key)
        if discovered:
            data = _try_models(discovered)

    if data is None:
        raise RuntimeError(last_error or "All Gemini model endpoints returned 404 Not Found.")

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates for evidence reasoning.")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    non_thought_parts = [part for part in parts if not part.get("thought")]
    selected_parts = non_thought_parts if non_thought_parts else parts
    text = "".join(str(part.get("text", "")) for part in selected_parts).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    if not text:
        raise ValueError("Gemini returned an empty reasoning response.")
    return json.loads(text)


def _call_openai_reasoner(api_key: str, payload_json: str) -> Dict[str, Any]:
    from openai import OpenAI

    model = get_env_setting("STOA_LLM_MODEL") or _DEFAULT_OPENAI_MODEL
    client = OpenAI(api_key=api_key, timeout=35.0, max_retries=1)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": _REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": payload_json},
        ],
        tools=[{
            "type": "function",
            "name": "submit_grounded_answer",
            "description": "Submit grounded reasoning decision over extracted drawing evidence.",
            "parameters": _REASONING_OPENAI_SCHEMA,
            "strict": True,
        }],
        tool_choice={"type": "function", "name": "submit_grounded_answer"},
        max_output_tokens=2048,
        store=False,
    )
    tool_call = next(
        (
            item for item in response.output
            if getattr(item, "type", None) == "function_call"
            and getattr(item, "name", None) == "submit_grounded_answer"
        ),
        None,
    )
    if tool_call is None:
        raise ValueError("OpenAI returned no grounded reasoning tool call.")
    return json.loads(tool_call.arguments)


def request_reasoning_decision(
    question: str,
    packs: Sequence[EvidencePack],
    *,
    history: Sequence[Dict[str, Any]] = (),
    forced_file_id: Optional[str] = None,
    twin_relationships: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Send structured EvidencePacks + question to Gemini/OpenAI and return (decision, provider)."""
    payload = {
        "question": question,
        "forced_file_id": forced_file_id,
        "recent_conversation": list(history[-6:]),
        "twin_relationships": twin_relationships or {},
        "files": [pack.catalog for pack in packs],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    g_key = gemini_api_key()
    if g_key:
        return _call_gemini_reasoner(g_key, payload_json), "gemini"

    o_key = openai_api_key()
    if o_key:
        return _call_openai_reasoner(o_key, payload_json), "openai"

    raise RuntimeError(
        "No LLM API key configured. Set GEMINI_API_KEY (or GOOGLE_API_KEY / OPENAI_API_KEY) "
        "in your environment or in a local .env file."
    )


def execute_reasoning_decision(
    decision: Dict[str, Any],
    packs_by_id: Dict[str, EvidencePack],
    *,
    provider: str = "gemini",
    fallback_options: Optional[List[Dict[str, Any]]] = None,
    question: str = "",
) -> Tuple[Optional[EvidencePack], Dict[str, Any]]:
    """Deterministically compute the answer & ground-truth citations from the LLM's decision."""
    action = str(decision.get("action") or "not_found")
    chosen_id = decision.get("file_id")
    pack = packs_by_id.get(str(chosen_id)) if chosen_id else (
        next(iter(packs_by_id.values()), None) if len(packs_by_id) == 1 else None
    )
    explanation = str(decision.get("explanation") or "").strip()

    if action == "clarification_required":
        clar_q = str(decision.get("clarification_question") or explanation or "Which drawing or scope should I use?").strip()
        res: Dict[str, Any] = {
            "status": "clarification_required",
            "answer": clar_q,
            "reason": explanation or "Additional detail is needed to verify the answer on the drawing.",
            "evidence": [],
        }
        if fallback_options:
            res["options"] = fallback_options
        return pack, res

    if action in {"not_found", "unsupported"}:
        return pack, {
            "status": action,
            "answer": None,
            "reason": explanation or "No verifiable evidence matching this question was found in the drawing.",
            "evidence": [],
        }

    if pack is None:
        return None, {
            "status": "not_found",
            "answer": None,
            "reason": explanation or "Could not match the selected file to an uploaded drawing.",
            "evidence": [],
        }

    raw_ids = [str(i).strip() for i in (decision.get("selected_ids") or []) if str(i).strip()]
    seen_ids: set[str] = set()
    selected_nodes: List[EvidenceNode] = []
    unknown_ids: List[str] = []
    for nid in raw_ids:
        if nid in seen_ids:
            continue
        seen_ids.add(nid)
        node = pack.nodes.get(nid)
        if node is None:
            # Also check if ID exists in another twin pack in case the LLM cited across packs
            other_pack = next((p for p in packs_by_id.values() if nid in p.nodes), None)
            if other_pack is not None:
                if not selected_nodes:
                    pack = other_pack
                if other_pack.file_id == pack.file_id:
                    selected_nodes.append(other_pack.nodes[nid])
                    continue
            unknown_ids.append(nid)
        else:
            selected_nodes.append(node)

    operation = str(decision.get("operation") or "direct_value")
    attribute = str(decision.get("attribute") or "none")
    unit = decision.get("unit")

    if not selected_nodes:
        # Allow zero count ONLY when the LLM explicitly performed count_selected on a verified empty category
        # and the underlying parser capability exists on that file!
        if operation == "count_selected" and not unknown_ids and pack.nodes:
            return pack, {
                "status": "answered",
                "value": 0,
                "answer": 0,
                "unit": unit or "count",
                "answer_text": "0",
                "reason": explanation or "0 matching items detected in the parsed evidence.",
                "method": f"llm_grounded_reasoning ({provider})",
                "review_recommended": pack.format == "image",
                "evidence": [{
                    "source": pack.catalog.get("file_name", pack.file_name),
                    "note": "Verified 0 matching instances in extracted specialist inventory.",
                }],
            }
        return pack, {
            "status": "not_found",
            "answer": None,
            "reason": (
                f"Rejected unverified evidence IDs ({', '.join(unknown_ids[:5])})."
                if unknown_ids
                else (explanation or "No grounded evidence items were cited to support an answer.")
            ),
            "evidence": [],
        }

    citations: List[Dict[str, Any]] = []
    for node in selected_nodes:
        citations.extend(node.citations)

    methods = list(dict.fromkeys(node.method for node in selected_nodes if node.method))
    combined_method = f"llm_grounded_reasoning ({provider}) + " + ", ".join(methods[:3])
    review_rec = any(node.review_recommended for node in selected_nodes)

    def _violates_geometric_count_guardrail() -> bool:
        if pack.format != "image" or not selected_nodes:
            return False
        if not any(n.kind in {"ocr_room_label", "ocr_text_line"} for n in selected_nodes):
            return False
        q_cf = question.casefold()
        label_only_cues = (
            "wc", "restroom", "toilet room", "powder room", "sauna", "löyly", "loyly",
            "parveke", "balcon", "terrace", "patio", "porch", "deck", "veranda", "loggia",
            "storey", "storeys", "kerros", "etage", "étage", "geschoss", "korrus", "label", "text",
        )
        if any(cue in q_cf for cue in label_only_cues):
            return False
        if operation == "count_selected":
            return True
        if unit == "count" or re.search(
            r"\b(?:how\s+many|number\s+of|count|kuinka\s+monta|montako|wie\s+viele|anzahl|combien|hvor\s+mange|mitu)\b",
            q_cf,
        ):
            return True
        return False

    if _violates_geometric_count_guardrail():
        return pack, {
            "status": "not_found",
            "answer": None,
            "reason": "Room and symbol counts on floor plans must be derived from geometric regions, not partial OCR text labels.",
            "evidence": [],
        }

    if operation == "count_selected":
        total_count = sum(int(node.count) for node in selected_nodes)
        return pack, {
            "status": "answered",
            "value": total_count,
            "answer": total_count,
            "unit": unit or "count",
            "answer_text": f"{total_count}",
            "reason": explanation,
            "method": combined_method,
            "review_recommended": review_rec,
            "evidence": citations,
        }

    if operation in {"sum_attribute", "min_attribute", "max_attribute"}:
        if attribute == "none":
            attribute = "net_floor_area_m2" if any("net_floor_area_m2" in n.attributes for n in selected_nodes) else "area_m2"
        values: List[Tuple[float, EvidenceNode]] = []
        for node in selected_nodes:
            if attribute in node.incomplete_attributes:
                return pack, {
                    "status": "not_found",
                    "answer": None,
                    "reason": f"Required quantity '{attribute}' is missing on {node.label}; abstaining rather than returning a partial value.",
                    "evidence": node.citations[:5],
                }
            val = node.attributes.get(attribute)
            if val is None and attribute == "area_m2":
                val = node.attributes.get("net_floor_area_m2") or node.attributes.get("side_area_m2")
            if val is None and attribute == "net_floor_area_m2":
                val = node.attributes.get("area_m2")
            if val is None or not math.isfinite(float(val)):
                return pack, {
                    "status": "not_found",
                    "answer": None,
                    "reason": f"Quantity '{attribute}' is unavailable on {node.label}.",
                    "evidence": node.citations[:5],
                }
            values.append((float(val), node))

        default_unit = "m2" if "area" in attribute else ("m3" if "volume" in attribute else "m")
        if operation == "sum_attribute":
            computed = round(sum(v for v, _ in values), 4)
            target_citations = citations
        elif operation == "max_attribute":
            best_v, best_node = max(values, key=lambda pair: pair[0])
            computed = round(best_v, 4)
            target_citations = best_node.citations
        else:
            best_v, best_node = min(values, key=lambda pair: pair[0])
            computed = round(best_v, 4)
            target_citations = best_node.citations

        return pack, {
            "status": "answered",
            "value": computed,
            "answer": computed,
            "unit": unit or default_unit,
            "answer_text": f"{computed} {unit or default_unit}",
            "reason": explanation,
            "method": combined_method,
            "review_recommended": review_rec,
            "evidence": target_citations,
        }

    # operation == "direct_value"
    direct_raw = decision.get("direct_answer")
    if direct_raw is None or not str(direct_raw).strip():
        if len(selected_nodes) == 1:
            direct_raw = selected_nodes[0].label
        else:
            direct_raw = explanation
    direct_str = str(direct_raw).strip()
    numeric_val: Any = direct_str
    if unit in {"count", "m2", "m", "m3", "mm"} and re.fullmatch(r"-?\d+(?:\.\d+)?", direct_str):
        numeric_val = int(direct_str) if unit == "count" and "." not in direct_str else float(direct_str)

    return pack, {
        "status": "answered",
        "value": numeric_val,
        "answer": numeric_val,
        "unit": unit,
        "answer_text": direct_str,
        "reason": explanation,
        "method": combined_method,
        "review_recommended": review_rec,
        "evidence": citations,
    }
