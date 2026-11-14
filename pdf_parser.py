"""Extract searchable, page-located text and schedule tables from PDFs."""

from __future__ import annotations

from pathlib import Path
import hashlib
import re
import tempfile
from typing import Any, Dict, List, Optional


def parse_pdf(file_path: str | Path) -> Dict[str, Any]:
    """Parse PDF text, words, tables, and page coordinates for evidence retrieval.

    Pages containing raster images and little embedded text receive shared
    RapidOCR extraction. OCR boxes are mapped back to PDF page coordinates.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got {path.suffix or 'no extension'}")
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("PDF parsing requires pdfplumber. Install it with: pip install pdfplumber") from exc

    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(x_tolerance=2, y_tolerance=2, use_text_flow=True)
            ocr_status = "not_needed" if len(words) >= 5 or not page.images else "skipped"
            if len(words) < 5 and page.images:
                try:
                    from image_parser import ocr_image_array
                    rendered = page.to_image(resolution=150).original.convert("RGB")
                    ocr_status, ocr_words = ocr_image_array(rendered)
                    if ocr_status == "available":
                        scale_x = float(page.width) / rendered.width
                        scale_y = float(page.height) / rendered.height
                        words = [{
                            "text": word["text"],
                            "x0": word["bbox"][0] * scale_x,
                            "top": word["bbox"][1] * scale_y,
                            "x1": word["bbox"][2] * scale_x,
                            "bottom": word["bbox"][3] * scale_y,
                            "confidence": word["confidence"],
                            "origin": "ocr",
                        } for word in ocr_words]
                except Exception as exc:
                    ocr_status = f"failed: {type(exc).__name__}: {exc}"
            table_objects = page.find_tables()
            tables = []
            for table in table_objects:
                rows = []
                extracted_rows = table.extract()
                for row_index, row in enumerate(table.rows):
                    extracted = extracted_rows[row_index]
                    rows.append({
                        "cells": [
                            {"text": cell_text, "bbox": list(cell_bbox) if cell_bbox else None}
                            for cell_text, cell_bbox in zip(extracted, row.cells)
                        ],
                    })
                tables.append({"page": page_number, "bbox": list(table.bbox), "rows": rows})
            pages.append({
                "page": page_number,
                "width": float(page.width),
                "height": float(page.height),
                "text": page.extract_text(x_tolerance=2, y_tolerance=2) or " ".join(w["text"] for w in words),
                "words": [
                    {key: word[key] for key in ("text", "x0", "top", "x1", "bottom", "confidence", "origin") if key in word}
                    for word in words
                ],
                "tables": tables,
                "image_count": len(page.images),
                "ocr_status": ocr_status,
                "vector_line_count": len(page.lines),
                "vector_rect_count": len(page.rects),
                "vector_curve_count": len(page.curves),
            })
    return {
        "source": {"path": str(path), "format": "PDF"},
        "page_count": len(pages),
        "pages": pages,
    }


_SCHEDULE_HEADER_WORDS = (
    "width", "height", "finish", "material", "matl", "fire rating",
    "number", "hardware", "area", "count", "qty", "quantity",
    "type mark", "level", "comments", "notes",
)


def _table_rows(table: Dict[str, Any]) -> List[List[Optional[str]]]:
    return [[cell.get("text") for cell in row.get("cells", [])] for row in table.get("rows", [])]


def _is_border_dump_table(table: Dict[str, Any]) -> bool:
    """Detect full-sheet title-block border rectangles mistaken for tables by pdfplumber."""
    rows = _table_rows(table)
    if not rows:
        return True
    max_len = max((len(str(cell or "")) for row in rows for cell in row), default=0)
    return max_len > 600


def _row_looks_like_header(row: List[Optional[str]]) -> bool:
    non_empty = [str(v).strip() for v in row if v is not None and str(v).strip()]
    if not non_empty:
        return False
    if max(len(v) for v in non_empty) > 80:
        return False
    joined = " ".join(non_empty).casefold()
    # Data rows with area units (e.g. "108 SF") are schedule records, not headers.
    if re.search(r"\b\d[\d,]*\s*(?:sf|sq\.?\s*ft|m²|m2)\b", joined):
        return False
    if any(word in joined for word in _SCHEDULE_HEADER_WORDS):
        return True
    if any(word in joined for word in ("north wall", "south wall", "east wall", "west wall")):
        return True
    return False


def _table_header(table: Dict[str, Any], linked_headers: List[Dict[str, Any]]) -> List[str]:
    if _is_border_dump_table(table):
        return []
    rows = _table_rows(table)
    if not rows:
        return []
    column_count = max(map(len, rows))
    header_rows = []
    for row in rows[:4]:
        if _row_looks_like_header(row):
            header_rows.append(row)
        elif header_rows:
            break
    if not header_rows:
        # Some drawings put the schedule headings in an adjacent table object directly above.
        row_count = max(map(len, rows))
        t_bbox = table.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        is_room_finish_table = any("SF" in str(cell or "") for cell in (rows[0] if rows else []))
        valid_candidates = [
            c for c in linked_headers
            if c is not table and not _is_border_dump_table(c) and _table_rows(c)
        ]

        def _cand_rank(candidate: Dict[str, Any]) -> tuple:
            c_rows = _table_rows(candidate)
            c_cols = max(map(len, c_rows))
            c_bbox = candidate.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            aligned_above = (
                abs(float(c_bbox[0]) - float(t_bbox[0])) < 35.0
                and -5.0 <= (float(t_bbox[1]) - float(c_bbox[3])) <= 65.0
            )
            c_text = " ".join(str(c or "") for r in c_rows for c in r).casefold()
            room_pref = 0 if (is_room_finish_table and "floor finish" in c_text) else 1
            return (
                0 if aligned_above else 1,
                abs(c_cols - row_count),
                room_pref,
                abs(float(t_bbox[1]) - float(c_bbox[3])),
            )

        for candidate in sorted(valid_candidates, key=_cand_rank):
            for row in _table_rows(candidate)[:4]:
                if _row_looks_like_header(row):
                    header_rows.append(row)
            if header_rows:
                break
    headers = []
    expanded_headers = []
    for row in header_rows:
        expanded = list(row)
        group = None
        for col, value in enumerate(row):
            if value and value.strip().casefold() in {"door", "frame", "window"}:
                group = value.strip()
            elif value and value.strip():
                # Keep group headers such as Door and Frame over their
                # subordinate Width/Height/Material columns.
                pass
            elif group and col < column_count:
                expanded[col] = group
        expanded_headers.append(expanded)
    for col in range(column_count):
        parts = []
        for row in expanded_headers:
            if col < len(row) and row[col] and row[col].strip() not in parts:
                parts.append(row[col].strip().replace("\n", " "))
        headers.append(" ".join(parts))
    return headers


def _expand_merged_schedule_rows(rows: List[List[Optional[str]]]) -> List[tuple[int, List[str]]]:
    """Unmerge adjacent schedule rows that lacked a horizontal ruling line in the PDF."""
    expanded: List[tuple[int, List[str]]] = []
    for r_idx, row in enumerate(rows):
        cells = [str(c or "").strip() for c in row]
        non_empty = [c for c in cells if c]
        if not non_empty:
            continue
        if r_idx < 4 and _row_looks_like_header(row):
            continue
        if r_idx == 0 and len(non_empty) == 1 and "schedule" in non_empty[0].casefold():
            continue
        line_counts = [len([ln for ln in c.splitlines() if ln.strip()]) for c in non_empty]
        if len(non_empty) >= 4 and len(set(line_counts)) == 1 and line_counts[0] >= 2:
            k = line_counts[0]
            split_cols = [
                [ln.strip() for ln in c.splitlines() if ln.strip()] if c else [""] * k
                for c in cells
            ]
            for sub_i in range(k):
                expanded.append((r_idx, [col[sub_i] for col in split_cols]))
        else:
            expanded.append((r_idx, [c.replace("\n", " ").strip() for c in cells]))
    return expanded


def _normal(text: Any) -> str:
    return "".join(ch.casefold() for ch in str(text or "") if ch.isalnum())


def _door_type_description(code: str, pages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for page in pages:
        words = page.get("words", [])
        labels = [w for w in words if str(w.get("text", "")).upper() == code.upper() and w.get("top", 0) < 700]
        for label in labels:
            same_line = [w for w in words if abs(w.get("top", 0) - label.get("top", 0)) < 2]
            if not any(str(w.get("text", "")).upper() == "DOOR" for w in same_line):
                continue
            centers = sorted(
                (w["x0"] + w["x1"]) / 2 for w in same_line
                if re.fullmatch(r"[A-Z]\d", str(w.get("text", "")).upper())
            )
            center = (label["x0"] + label["x1"]) / 2
            previous = max((value for value in centers if value < center), default=center - 220)
            following = min((value for value in centers if value > center), default=center + 220)
            left, right = (previous + center) / 2, (center + following) / 2
            description_words = [
                w for w in words
                if label["top"] + 5 < w.get("top", 0) < label["top"] + 45
                and left <= (w["x0"] + w["x1"]) / 2 < right
            ]
            description = " ".join(w["text"] for w in sorted(description_words, key=lambda item: (item["top"], item["x0"])))
            if description:
                if "sliding" in description.casefold() and "door" not in description.casefold():
                    description += " door"
                bbox = [min(w["x0"] for w in description_words), min(w["top"] for w in description_words),
                        max(w["x1"] for w in description_words), max(w["bottom"] for w in description_words)]
                return {"page": page.get("page", 1), "bbox": bbox, "text": description}
    return None


def _find_row(tables: List[Dict[str, Any]], target: str, category: str) -> Optional[Dict[str, Any]]:
    target_norm = _normal(target)
    choices = []
    for table in tables:
        if _is_border_dump_table(table):
            continue
        raw_rows = _table_rows(table)
        title = " ".join(str(cell or "") for row in raw_rows[:3] for cell in row).casefold()
        for raw_idx, row in _expand_merged_schedule_rows(raw_rows):
            if not any(_normal(cell) == target_norm for cell in row if cell):
                continue
            if len(row) <= 1:
                continue
            if category == "door" and "door" not in title and "mark" not in title:
                continue
            if category == "room" and "floor finish" not in title and "room finish" not in title:
                # The heading can be a separate table object; room data rows
                # have an area in the first cell and a room name in the third.
                if len(row) < 8 or not any("SF" in str(cell or "") for cell in row[:1]):
                    continue
            choices.append((table, raw_idx, row, title))
    best = max(choices, key=lambda item: item[0].get("bbox", [0, 0, 0, 0])[1], default=None)
    if not best:
        return None
    table, index, row, title = best
    return {"table": table, "index": index, "row": row, "title": title}


def _page_dimensions(pages: List[Dict[str, Any]], page_num: int) -> Optional[List[float]]:
    for p in pages:
        if p.get("page") == page_num and p.get("width") and p.get("height"):
            return [float(p["width"]), float(p["height"])]
    return None


def _attach_pdf_page_sizes(result: Dict[str, Any], pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    for item in result.get("evidence") or []:
        if isinstance(item, dict):
            page_num = int(item.get("page", 1) or 1)
            dims = _page_dimensions(pages, page_num)
            if dims and "page_size" not in item:
                item["page_size"] = dims
            if "bbox" not in item and item.get("table_bbox"):
                item["bbox"] = item["table_bbox"]
            item.setdefault("coordinate_system", "PDF points; origin at top-left")
    return result


def _norm_dim_text(text: str) -> str:
    s = str(text or "").strip().casefold()
    s = s.replace("''", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"'\s*-\s*", "'", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_area_cell(text: str) -> Optional[tuple[float | int, str]]:
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*(sf|sq\.?\s*ft\.?|m²|m2|sq\.?\s*m\.?)\b", str(text or ""), re.I)
    if not m:
        return None
    raw_num = float(m.group(1).replace(",", ""))
    num: float | int = int(raw_num) if raw_num.is_integer() else round(raw_num, 4)
    unit_tok = m.group(2).casefold()
    unit = "ft2" if ("sf" in unit_tok or "ft" in unit_tok) else "m2"
    return num, unit


def _extract_structured_schedules(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract normalized room, door, and window schedule tables with headers and logical rows."""
    schedules: List[Dict[str, Any]] = []
    for page in pages:
        page_num = int(page.get("page", 1) or 1)
        page_tables = [t for t in (page.get("tables") or []) if not _is_border_dump_table(t)]
        for table in page_tables:
            raw_rows = _table_rows(table)
            if len(raw_rows) < 2:
                continue
            logical_rows = _expand_merged_schedule_rows(raw_rows)
            if not logical_rows:
                continue
            headers = _table_header(table, page_tables)
            if not any(headers):
                continue
            h_cf = [h.casefold() for h in headers]

            title_parts: List[str] = []
            r0_non_empty = [str(c).strip() for c in raw_rows[0] if c and str(c).strip()]
            if len(r0_non_empty) == 1:
                title_parts.append(r0_non_empty[0])
            t_bbox = table.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            for cand in page_tables:
                if cand is table:
                    continue
                c_bbox = cand.get("bbox") or [0.0, 0.0, 0.0, 0.0]
                if abs(float(c_bbox[0]) - float(t_bbox[0])) < 35.0 and -5.0 <= (float(t_bbox[1]) - float(c_bbox[3])) <= 65.0:
                    c_rows = _table_rows(cand)
                    if c_rows:
                        c0 = [str(c).strip() for c in c_rows[0] if c and str(c).strip()]
                        if len(c0) == 1:
                            title_parts.append(c0[0])
            title_cf = " ".join(title_parts).casefold()

            if any("floor finish" in h or "ceiling finish" in h or "wall finish" in h for h in h_cf) or (
                any("room" in h for h in h_cf) and any("area" in h for h in h_cf)
            ):
                kind = "room_schedule"
            elif "window" in title_cf or any("window" in h for h in h_cf):
                kind = "window_schedule"
            elif "door" in title_cf or any(
                k in h for h in h_cf for k in ("door", "frame", "hardware", "fire rating")
            ):
                kind = "door_schedule"
            else:
                continue

            count_col: Optional[int] = None
            for col_i, h in enumerate(h_cf):
                if h in {"count", "qty", "quantity"} or h.endswith(" count"):
                    vals = [r[col_i] for _, r in logical_rows if col_i < len(r) and r[col_i]]
                    if vals and sum(1 for v in vals if re.fullmatch(r"\d+", v)) >= max(1, len(vals) // 2):
                        count_col = col_i
                        break

            schedules.append({
                "kind": kind,
                "title": title_cf,
                "page": page_num,
                "table": table,
                "bbox": table.get("bbox"),
                "headers": headers,
                "headers_cf": h_cf,
                "count_col": count_col,
                "rows": logical_rows,
            })
    return schedules


def _answer_pdf_schedule_aggregate(question: str, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Answer generic count and area questions over structured PDF room, door, and window schedules."""
    pages = document.get("pages", [])
    source_path = document.get("source", {}).get("path")
    schedules = _extract_structured_schedules(pages)
    if not schedules:
        return None

    # Strip leading sheet/project locator preamble ("On sheet X of Y drawings (...),") so
    # sheet IDs, addresses, and project names never interfere with schedule column filters.
    q_clause = question.strip()
    m_how = re.search(r"\b(if\s+every\b.*|how\s+many\b.*|what\s+is\s+the\s+(?:total\s+|combined\s+)?area\b.*)", question, re.I)
    if m_how:
        q_clause = m_how.group(1).strip()
    else:
        q_clause = re.sub(r"^\s*on\s+sheet\s+\S+[^,]*,\s*", "", q_clause, flags=re.I)
    q_cf = q_clause.casefold()

    is_count = any(k in q_cf for k in ("how many", "number of", "counted", "count", "needed in total"))
    is_area = any(k in q_cf for k in ("area", "square feet", "sq ft", "sf"))

    def _make_schedule_result(
        sched: Dict[str, Any],
        selected_rows: List[tuple[int, List[str]]],
        value: int | float,
        unit: str,
    ) -> Dict[str, Any]:
        raw_indices = sorted({idx for idx, _ in selected_rows})
        return _attach_pdf_page_sizes({
            "status": "answered",
            "value": value,
            "answer": value,
            "unit": unit,
            "method": "pdf_schedule_table",
            "evidence": [{
                "page": sched["page"],
                "bbox": sched["bbox"],
                "table_bbox": sched["bbox"],
                "headers": sched["headers"],
                "rows": raw_indices,
                "rows_text": [r for _, r in selected_rows[:30]],
                "source": source_path,
            }],
        }, pages)

    # 1. Room Finish / Area Schedule queries
    room_schedules = [s for s in schedules if s["kind"] == "room_schedule"]
    if room_schedules and (is_area or (is_count and "room" in q_cf and "door" not in q_cf)):
        sched = room_schedules[0]
        h_cf = sched["headers_cf"]
        area_col = next((i for i, h in enumerate(h_cf) if "area" in h), 0)
        num_col = next((i for i, h in enumerate(h_cf) if "number" in h), 1)
        name_col = next((i for i, h in enumerate(h_cf) if "name" in h), 2)
        floor_col = next((i for i, h in enumerate(h_cf) if "floor finish" in h), None)
        base_col = next((i for i, h in enumerate(h_cf) if "base" in h), None)
        ceil_col = next((i for i, h in enumerate(h_cf) if "ceiling" in h), None)
        wall_cols = [i for i, h in enumerate(h_cf) if "wall" in h]
        finish_cols = [c for c in [floor_col, base_col, ceil_col, *wall_cols] if c is not None]

        valid_rows = [
            (idx, r) for idx, r in sched["rows"]
            if (num_col < len(r) and r[num_col]) or (name_col < len(r) and r[name_col])
        ]

        # 1a. Single room area lookup
        if is_area and not is_count and "total" not in q_cf and "combined" not in q_cf:
            m_rm = re.search(r"\broom\s+([a-z0-9.-]+)\b", q_cf)
            if m_rm:
                target_rm = _normal(m_rm.group(1))
                for idx, r in valid_rows:
                    if num_col < len(r) and _normal(r[num_col]) == target_rm and area_col < len(r):
                        parsed_area = _parse_area_cell(r[area_col])
                        if parsed_area is not None:
                            val, unit = parsed_area
                            return _make_schedule_result(sched, [(idx, r)], val, unit)

        # 1b. All four walls finished with <CODE>
        if ("all four walls" in q_cf or "all walls" in q_cf) and len(wall_cols) >= 4:
            q_codes = re.findall(r"\b([A-Z]+\d+)\b", q_clause)
            if q_codes:
                code = q_codes[0].upper()
                matched = [
                    (idx, r) for idx, r in valid_rows
                    if all(wc < len(r) and r[wc].strip().upper() == code for wc in wall_cols[:4])
                ]
                if is_count and not is_area:
                    return _make_schedule_result(sched, matched, len(matched), "count")

        # 1c. Filter by finish code (Floor / Base / Ceiling / Wall)
        finish_vocab = {
            tok.upper()
            for _, r in valid_rows
            for fc in finish_cols
            if fc < len(r)
            for tok in re.findall(r"\b[A-Z]+\d+\b", r[fc].upper())
        }
        q_finish_codes = [c.upper() for c in re.findall(r"\b([A-Z]+\d+)\b", q_clause) if c.upper() in finish_vocab]
        if q_finish_codes:
            target_cols = (
                [floor_col] if "floor" in q_cf and floor_col is not None else
                [base_col] if "base" in q_cf and base_col is not None else
                [ceil_col] if "ceiling" in q_cf and ceil_col is not None else
                finish_cols
            )
            matched = [
                (idx, r) for idx, r in valid_rows
                if any(
                    tc < len(r) and any(re.search(rf"\b{re.escape(code)}\b", r[tc].upper()) for code in q_finish_codes)
                    for tc in target_cols
                )
            ]
            if is_count and not is_area:
                return _make_schedule_result(sched, matched, len(matched), "count")
            if is_area and not is_count:
                areas = [_parse_area_cell(r[area_col]) for _, r in matched if area_col < len(r)]
                areas = [a for a in areas if a is not None]
                if areas:
                    total_a = sum(a[0] for a in areas)
                    return _make_schedule_result(sched, matched, total_a, areas[0][1])

        # 1d. Total / combined area of all rooms or rooms matching room name
        if is_area and not is_count:
            m_named = re.search(r"\bnamed\s+([A-Za-z0-9/&' -]+?)(?:\s+in\s+the|\s+on\s+sheet|[?.!]|$)", q_clause, re.I)
            if m_named:
                wanted_name = m_named.group(1).strip().casefold()
                matched = [
                    (idx, r) for idx, r in valid_rows
                    if name_col < len(r) and r[name_col].strip().casefold() == wanted_name
                ]
            elif "all rooms" in q_cf:
                matched = [
                    (idx, r) for idx, r in valid_rows
                    if area_col < len(r) and _parse_area_cell(r[area_col]) is not None
                ]
            else:
                name_words = {
                    tok
                    for _, r in valid_rows
                    if name_col < len(r)
                    for tok in re.findall(r"[a-z]{3,}", r[name_col].casefold())
                    if tok not in {"room", "rooms", "space", "see", "plan"}
                }
                q_words = set(re.findall(r"[a-z]{3,}", q_cf))
                matched_words = {
                    nw for nw in name_words
                    if nw in q_words or f"{nw}s" in q_words or nw.rstrip("s") in q_words
                }
                matched = [
                    (idx, r) for idx, r in valid_rows
                    if matched_words and name_col < len(r)
                    and any(mw in r[name_col].casefold() for mw in matched_words)
                ]
            if matched:
                areas = [_parse_area_cell(r[area_col]) for _, r in matched if area_col < len(r)]
                areas = [a for a in areas if a is not None]
                if areas:
                    total_a = sum(a[0] for a in areas)
                    return _make_schedule_result(sched, matched, total_a, areas[0][1])

        # 1e. Unfiltered room count in finish room schedule
        if is_count and not is_area and "schedule" in q_cf and ("listed" in q_cf or "in total" in q_cf or "are in" in q_cf):
            return _make_schedule_result(sched, valid_rows, len(valid_rows), "count")

    # 2. Door & Window Schedule count queries
    if not is_count:
        return None
    wants_window = "window" in q_cf and "door" not in q_cf
    target_kind = "window_schedule" if wants_window else "door_schedule"
    cand_schedules = [s for s in schedules if s["kind"] == target_kind]
    if not cand_schedules:
        return None

    def _row_qty(sched: Dict[str, Any], row: List[str]) -> int:
        cc = sched["count_col"]
        if cc is not None and cc < len(row):
            m_int = re.search(r"\d+", row[cc])
            return int(m_int.group(0)) if m_int else 0
        return 1

    # Prefer summary vs instance schedule when explicitly requested
    if len(cand_schedules) > 1:
        if "door type schedule" in q_cf:
            summary_s = [s for s in cand_schedules if s["count_col"] is not None]
            if summary_s:
                cand_schedules = summary_s
        elif "door numbers" in q_cf:
            inst_s = [s for s in cand_schedules if s["count_col"] is None]
            if inst_s:
                cand_schedules = inst_s
        else:
            # Prefer summary schedule first when "scheduled" is used, instance when "door schedule"
            cand_schedules = sorted(
                cand_schedules,
                key=lambda s: (0 if (s["count_col"] is not None) == ("scheduled" in q_cf and "door schedule" not in q_cf) else 1),
            )

    # 2a. N-wide assembly unit multiplication
    m_mult = re.search(r"every\s+n-wide\b.*?\bhow\s+many\s+([A-Za-z0-9-]+)\s+units", q_clause, re.I)
    if m_mult:
        model_tok = m_mult.group(1).casefold()
        for sched in cand_schedules:
            matched = [(idx, r) for idx, r in sched["rows"] if model_tok in " ".join(r).casefold()]
            if matched:
                total_units = 0
                for _, r in matched:
                    row_txt = " ".join(r)
                    m_w = re.search(r"(\d+)\s*-\s*wide", row_txt, re.I)
                    mult = int(m_w.group(1)) if m_w else 1
                    total_units += _row_qty(sched, r) * mult
                return _make_schedule_result(sched, matched, total_units, "count")

    # 2b. Prefix match ("starts with 'SWING'")
    m_pref = re.search(r"starts\s+with\s+['\"]([^'\"]+)['\"]", q_clause, re.I)
    if m_pref:
        prefix = m_pref.group(1).strip().casefold()
        for sched in cand_schedules:
            desc_cols = [
                i for i, h in enumerate(sched["headers_cf"])
                if h in {"type", "description"} or "comments" in h
            ] or list(range(len(sched["headers"])))
            matched = [
                (idx, r) for idx, r in sched["rows"]
                if any(dc < len(r) and r[dc].strip().casefold().startswith(prefix) for dc in desc_cols)
            ]
            if matched:
                return _make_schedule_result(sched, matched, sum(_row_qty(sched, r) for _, r in matched), "count")

    # 2c. Note code filter ("carry note A")
    m_note = re.search(r"\bnote\s+([A-Za-z0-9]+)\b", q_clause, re.I)
    if m_note:
        note_code = m_note.group(1).upper()
        for sched in cand_schedules:
            note_cols = [i for i, h in enumerate(sched["headers_cf"]) if "note" in h or "comment" in h]
            if not note_cols:
                continue
            matched = [
                (idx, r) for idx, r in sched["rows"]
                if any(
                    nc < len(r)
                    and note_code in [tok.strip() for tok in re.split(r"[,;/&]+", r[nc].upper()) if tok.strip()]
                    for nc in note_cols
                )
            ]
            if matched:
                return _make_schedule_result(sched, matched, sum(_row_qty(sched, r) for _, r in matched), "count")

    # 2d. Room number filter ("for room 118")
    m_room = re.search(r"\broom\s+(\d+[A-Za-z]?)\b", q_clause, re.I)
    if m_room:
        rm_norm = _normal(m_room.group(1))
        for sched in cand_schedules:
            rm_cols = [i for i, h in enumerate(sched["headers_cf"]) if "room" in h and "number" in h]
            if not rm_cols:
                continue
            matched = [
                (idx, r) for idx, r in sched["rows"]
                if any(rc < len(r) and _normal(r[rc]) == rm_norm for rc in rm_cols)
            ]
            if matched:
                return _make_schedule_result(sched, matched, sum(_row_qty(sched, r) for _, r in matched), "count")

    # 2e. Explicit type / type-mark codes ("door type A1", "type U01", "types SS, TT and UU")
    m_types = re.search(
        r"\b(?:[Dd]oor\s+)?[Tt]ypes?\s+([A-Z0-9-]+(?:\s*(?:,|\band\b|\bor\b)\s*[A-Z0-9-]+)*)",
        q_clause,
    )
    if m_types:
        raw_list = re.split(r"\s*(?:,|\band\b|\bor\b)\s*", m_types.group(1).strip())
        wanted_codes = {c.strip().upper() for c in raw_list if c.strip()}
        for sched in cand_schedules:
            type_cols = [
                i for i, h in enumerate(sched["headers_cf"])
                if "type mark" in h or "door type" in h or "door door type" in h or h == "mark"
            ]
            if not type_cols:
                continue
            present_codes = {
                r[tc].strip().upper()
                for _, r in sched["rows"]
                for tc in type_cols
                if tc < len(r) and r[tc].strip()
            }
            if wanted_codes.issubset(present_codes):
                matched = [
                    (idx, r) for idx, r in sched["rows"]
                    if any(tc < len(r) and r[tc].strip().upper() in wanted_codes for tc in type_cols)
                ]
                return _make_schedule_result(sched, matched, sum(_row_qty(sched, r) for _, r in matched), "count")

    # 2f. Width / Height dimension filter ("3'-0\" wide", "7'-0\" high")
    m_dim = re.search(r"(\d+\'\s*-?\s*\d+(?:\s+\d+/\d+)?\"|\d+(?:\s+\d+/\d+)?\")\s*(wide|high)\b", q_clause, re.I)
    if m_dim:
        wanted_dim = _norm_dim_text(m_dim.group(1))
        axis = "width" if m_dim.group(2).casefold() == "wide" else "height"
        for sched in cand_schedules:
            dim_cols = [i for i, h in enumerate(sched["headers_cf"]) if axis in h]
            if not dim_cols:
                continue
            matched = [
                (idx, r) for idx, r in sched["rows"]
                if any(dc < len(r) and _norm_dim_text(r[dc]) == wanted_dim for dc in dim_cols)
            ]
            if matched:
                return _make_schedule_result(sched, matched, sum(_row_qty(sched, r) for _, r in matched), "count")

    # 2g. Material / Frame code filter ("material WD-1", "STL-1", "AL-1 or AL-2")
    mat_codes = [c.upper() for c in re.findall(r"\b[A-Z]{2,4}-\d+\b", q_clause)]
    if mat_codes:
        wants_frame = "frame" in q_cf
        for sched in cand_schedules:
            mat_cols = [
                i for i, h in enumerate(sched["headers_cf"])
                if (
                    ("frame" in h and ("mat" in h or "type" in h))
                    if wants_frame
                    else (("door" in h and "mat" in h) or ("material" in h and "frame" not in h))
                )
            ]
            if not mat_cols:
                continue
            matched = [
                (idx, r) for idx, r in sched["rows"]
                if any(
                    mc < len(r) and any(code in re.findall(r"[A-Z0-9-]+", r[mc].upper()) for code in mat_codes)
                    for mc in mat_cols
                )
            ]
            if matched:
                return _make_schedule_result(sched, matched, sum(_row_qty(sched, r) for _, r in matched), "count")

    # 2h. Unfiltered total when explicitly asking for all scheduled items/rows
    if any(
        phrase in q_cf
        for phrase in (
            "all type marks",
            "listed in the door schedule",
            "listed in the window schedule",
        )
    ):
        sched = cand_schedules[0]
        return _make_schedule_result(sched, sched["rows"], sum(_row_qty(sched, r) for _, r in sched["rows"]), "count")

    # 2i. Descriptive / floor phrase filter against schedule's distinctive row vocabulary
    stop_words = {
        "how", "many", "are", "there", "the", "and", "with", "for", "from", "that", "this",
        "door", "doors", "window", "windows", "type", "types", "mark", "marks", "wide", "high",
        "item", "items", "total", "count", "counted", "listed", "schedule", "scheduled",
        "including", "gate", "gates", "access", "panel", "panels", "clear", "opening", "openings",
        "unit", "units", "assembly", "assemblies", "number", "numbers", "floor", "leaves",
    }
    floor_match = re.search(r"\b(1st\s+floor|2nd\s+floor|3rd\s+floor|4th\s+floor|basement)\b", q_cf)
    floor_phrase = re.sub(r"\s+", " ", floor_match.group(1)) if floor_match else None
    q_desc_words = {w for w in re.findall(r"[a-z]{4,}", q_cf) if w not in stop_words}

    for sched in cand_schedules:
        n_rows = max(1, len(sched["rows"]))
        word_df: Dict[str, int] = {}
        for _, r in sched["rows"]:
            row_words = set(re.findall(r"[a-z]{4,}", " ".join(r).casefold())) - stop_words
            for w in row_words:
                word_df[w] = word_df.get(w, 0) + 1
        distinctive_vocab = {w for w, df in word_df.items() if df < 0.85 * n_rows}
        matched_vocab = q_desc_words & distinctive_vocab
        if not matched_vocab and not floor_phrase:
            continue
        matched = []
        for idx, r in sched["rows"]:
            row_joined = re.sub(r"\s+", " ", " ".join(r).casefold())
            row_words = set(re.findall(r"[a-z]{4,}", row_joined))
            if matched_vocab and not matched_vocab.issubset(row_words):
                continue
            if floor_phrase and floor_phrase not in row_joined:
                continue
            matched.append((idx, r))
        if matched:
            return _make_schedule_result(sched, matched, sum(_row_qty(sched, r) for _, r in matched), "count")

    return None


def _answer_pdf_text_lookup(question: str, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Answer title-block, quoted-label, or printed-dimension questions from PDF words."""
    from image_parser import _answer_raster_text_question, _fold_text, _ocr_lines, _drawing_metadata, _DIMENSION_PATTERN

    folded = _fold_text(question)
    aggregate_terms = (
        "how many", "number of", "count", "quantity of", "combien", "wie viele",
        "anzahl", "kuinka monta", "montako", "hvor mange", "mitu", "area", "flache",
        "surface", "superficie", "pinta-ala", "pindala", "volume", "volumen",
    )
    if any(term in folded for term in aggregate_terms):
        return None

    source_path = document.get("source", {}).get("path")
    for page in document.get("pages", []):
        raw_words = page.get("words", [])
        if not raw_words:
            continue
        word_boxes = [
            {
                "text": str(w.get("text", "")),
                "confidence": float(w.get("confidence", 1.0)),
                "bbox": [float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"])],
            }
            for w in raw_words
            if str(w.get("text", "")).strip() and all(k in w for k in ("x0", "top", "x1", "bottom"))
        ]
        if not word_boxes:
            continue
        lines = _ocr_lines(word_boxes)
        measurements = [
            {"text": m.group(0), "bbox": line["bbox"], "confidence": line["confidence"], "page": page["page"]}
            for line in lines for m in _DIMENSION_PATTERN.finditer(line["text"])
        ]
        synthetic_doc = {
            "source": {"path": source_path, "format": "PDF"},
            "lines": lines,
            "measurements": measurements,
            "drawing_metadata": _drawing_metadata(lines),
        }
        res = _answer_raster_text_question(question, synthetic_doc)
        if res is not None and res.get("status") == "answered":
            for ev in res.get("evidence") or []:
                ev["page"] = page["page"]
                ev["page_size"] = [float(page["width"]), float(page["height"])]
                ev["coordinate_system"] = "PDF points; origin at top-left"
            if res.get("method"):
                res["method"] = str(res["method"]).replace("raster_ocr_", "pdf_text_")
            return res
    return None


def _render_pdf_page_document(
    pdf_path: Path,
    page_info: Dict[str, Any],
    document: Dict[str, Any],
    render_dir: Optional[Path] = None,
    resolution: int = 150,
) -> Dict[str, Any]:
    """Render one PDF page to PNG once and cache its parsed image document."""
    page_num = int(page_info.get("page", 1))
    rendered_cache = document.setdefault("rendered_pages", {})
    if page_num in rendered_cache:
        return rendered_cache[page_num]

    import pdfplumber
    from image_parser import parse_image

    target_dir = Path(render_dir) if render_dir else Path(tempfile.gettempdir()) / "stoa-pdf-renders"
    target_dir.mkdir(parents=True, exist_ok=True)
    stat = pdf_path.stat()
    digest = hashlib.sha256(
        f"{pdf_path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\0p{page_num}\0dpi{resolution}".encode("utf-8")
    ).hexdigest()[:16]
    image_path = target_dir / f"{pdf_path.stem}_p{page_num}_{digest}.png"
    if not image_path.is_file():
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            rendered = page.to_image(resolution=resolution).original.convert("RGB")
            rendered.save(image_path, format="PNG")

    image_doc = parse_image(image_path, run_ocr=True)
    # Classify rendered page with CLIP if available so elevations/schedules are distinguished from floor plans.
    drawing_type = "uncertain"
    try:
        from file_classifier import DocumentRouter
        router = DocumentRouter(str(pdf_path.parent))
        predicted = router._classify_image(image_path)
        drawing_type = predicted or "uncertain"
    except Exception:
        drawing_type = "uncertain"
    image_doc["drawing_type"] = drawing_type
    image_doc["pdf_page"] = page_num
    image_doc["pdf_page_size"] = [float(page_info["width"]), float(page_info["height"])]
    rendered_cache[page_num] = image_doc
    return image_doc


def _answer_pdf_visual_fallback(
    question: str,
    document: Dict[str, Any],
    *,
    render_dir: Optional[Path] = None,
    model_path: Optional[str | Path] = None,
    geometry_model_path: Optional[str | Path] = None,
    cubicasa_repo_path: Optional[str | Path] = None,
    cubicasa_weights_path: Optional[str | Path] = None,
    object_detector_model: Optional[str | Path] = None,
    query_plan: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Render a PDF drawing page when visual/floor-plan analysis is needed."""
    q_low = question.casefold()
    if any(
        term in q_low
        for term in (
            "schedule", "scheduled", "in the model", "im modell",
            "dans le modele", "dans le modèle", "enlarged plan", "detail ",
        )
    ):
        return None
    source_str = document.get("source", {}).get("path")
    if not source_str:
        return None
    pdf_path = Path(source_str)
    if not pdf_path.is_file():
        return None
    pages = document.get("pages", [])
    if not pages:
        return None

    page_match = re.search(r"\b(?:page|sivu|seite|blad)\s+(\d+)\b", q_low)
    selected_page: Optional[Dict[str, Any]] = None
    if page_match:
        wanted_page = int(page_match.group(1))
        selected_page = next((p for p in pages if p.get("page") == wanted_page), None)
        if selected_page is None:
            return {
                "status": "not_found",
                "answer": None,
                "reason": f"Page {wanted_page} does not exist in this {len(pages)}-page PDF.",
                "evidence": [],
            }
    elif len(pages) == 1:
        selected_page = pages[0]
    elif len(pages) <= 5:
        drawing_pages = [
            p for p in pages
            if (p.get("vector_line_count", 0) + p.get("vector_curve_count", 0)) >= 20 or p.get("image_count", 0) >= 1
        ]
        if len(drawing_pages) == 1:
            selected_page = drawing_pages[0]
        elif len(drawing_pages) > 1:
            return {
                "status": "clarification_required",
                "answer": f"This PDF has {len(pages)} pages; specify the page number (e.g. 'on page 1') for visual analysis.",
                "reason": "Visual floor-plan analysis requires selecting a specific page on multi-sheet PDFs.",
                "evidence": [],
            }
    if selected_page is None:
        return None

    image_doc = _render_pdf_page_document(pdf_path, selected_page, document, render_dir=render_dir)
    from image_parser import answer_image_question
    visual_res = answer_image_question(
        question,
        image_doc,
        model_path=model_path,
        geometry_model_path=geometry_model_path,
        cubicasa_repo_path=cubicasa_repo_path,
        cubicasa_weights_path=cubicasa_weights_path,
        object_detector_model=object_detector_model,
        query_plan=query_plan,
    )
    if visual_res.get("status") == "unsupported":
        return None

    page_num = int(selected_page.get("page", 1))
    page_w, page_h = float(selected_page["width"]), float(selected_page["height"])
    img_w = max(1.0, float(image_doc.get("width") or page_w))
    img_h = max(1.0, float(image_doc.get("height") or page_h))
    sx, sy = page_w / img_w, page_h / img_h

    mapped_evidence = []
    for ev in visual_res.get("evidence") or []:
        ev_copy = dict(ev)
        ev_copy["source"] = str(pdf_path)
        ev_copy["page"] = page_num
        ev_copy["page_size"] = [page_w, page_h]
        raw_bbox = ev_copy.get("bbox")
        if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
            ev_copy["bbox"] = [
                round(float(raw_bbox[0]) * sx, 2),
                round(float(raw_bbox[1]) * sy, 2),
                round(float(raw_bbox[2]) * sx, 2),
                round(float(raw_bbox[3]) * sy, 2),
            ]
        ev_copy["coordinate_system"] = "PDF points; origin at top-left"
        mapped_evidence.append(ev_copy)

    out = dict(visual_res)
    out["evidence"] = mapped_evidence
    if out.get("method"):
        out["method"] = f"pdf_rendered_visual:{out['method']}"
    out["review_recommended"] = True
    return out


def answer_pdf_question(
    question: str,
    document: Dict[str, Any],
    *,
    enable_visual_fallback: bool = False,
    render_dir: Optional[Path] = None,
    model_path: Optional[str | Path] = None,
    geometry_model_path: Optional[str | Path] = None,
    cubicasa_repo_path: Optional[str | Path] = None,
    cubicasa_weights_path: Optional[str | Path] = None,
    object_detector_model: Optional[str | Path] = None,
    query_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieve a grounded schedule, text, or rendered-visual answer from a PDF."""
    pages = document.get("pages", [])
    all_text = "\n".join(page.get("text", "") for page in pages)
    tables = [table for page in pages for table in page.get("tables", [])]
    door_match = re.search(r"door\s+(\d+[a-z]?\.[a-z0-9]+)", question, re.I)
    if not door_match:
        door_match = re.search(r"(?:doors?\s*\(type\s+|type\s+)(u\d+)\b", question, re.I)
    if not door_match:
        door_match = re.search(r"(\d+\.[a-z]\d+)\s*-\s*\d+\.[a-z]\d+", question, re.I)
    room_match = re.search(r"room\s+(\d+)\b", question, re.I)
    target = door_match.group(1) if door_match else room_match.group(1) if room_match else None
    category = "door" if door_match else "room" if room_match else ""
    evidence: List[Dict[str, Any]] = []

    if "how many rooms have floor finish" in question.casefold():
        header_table = next(
            (t for t in tables if not _is_border_dump_table(t) and any("Floor Finish" in str(c or "") for r in _table_rows(t) for c in r)),
            None,
        )
        data_table = None
        for table in tables:
            if _is_border_dump_table(table):
                continue
            rows = _table_rows(table)
            if any(len(row) >= 10 and any("SF" in str(cell or "") for cell in row[:1]) for row in rows):
                data_table = table
        if data_table and header_table:
            headers = _table_header(header_table, [])
            finish_col = next((i for i, h in enumerate(headers) if "floor finish" in h.casefold()), 3)
            m_fcode = re.search(r"floor\s+finish\s+([A-Z]+\d+)\b", question, re.I)
            wanted_fcode = m_fcode.group(1).upper() if m_fcode else "F0"
            area_values = []
            selected = []
            selected_rows_text = []
            for row_index, row in _expand_merged_schedule_rows(_table_rows(data_table)):
                if len(row) > finish_col and re.search(rf"\b{re.escape(wanted_fcode)}\b", str(row[finish_col] or "").upper()):
                    match = re.search(r"([\d,]+)\s*SF", str(row[0] or ""), re.I)
                    if match:
                        area_values.append(int(match.group(1).replace(",", "")))
                    selected.append(row_index)
                    selected_rows_text.append(row)
            if selected:
                answer = f"{len(selected_rows_text)} rooms, {sum(area_values)} SF"
                return _attach_pdf_page_sizes({"status": "answered", "answer": answer, "evidence": [{
                    "page": 1, "table_bbox": data_table.get("bbox"), "rows": sorted(set(selected)),
                    "rows_text": selected_rows_text,
                    "source": document.get("source", {}).get("path"),
                }]}, pages)

    if "largest area" in question.casefold():
        data_table = next(
            (t for t in tables if not _is_border_dump_table(t) and any(len(row) >= 10 and any("SF" in str(c or "") for c in row[:1]) for row in _table_rows(t))),
            None,
        )
        if data_table:
            candidates = []
            for i, row in _expand_merged_schedule_rows(_table_rows(data_table)):
                if len(row) < 3:
                    continue
                match = re.search(r"([\d,]+)\s*SF", str(row[0] or ""), re.I)
                if match:
                    candidates.append((int(match.group(1).replace(",", "")), i, row))
            if candidates:
                area, row_index, row = max(candidates)
                answer = f"Room {row[1]} {row[2]} ({area} SF)"
                return _attach_pdf_page_sizes({"status": "answered", "answer": answer, "evidence": [{
                    "page": 1, "table_bbox": data_table.get("bbox"), "rows": [row_index],
                    "rows_text": [row], "source": document.get("source", {}).get("path"),
                }]}, pages)

    if target and "how many" not in question.casefold():
        found = _find_row(tables, target, category)
        if found:
            row = found["row"]
            header_tables = [t for t in tables if any("width" in str(c or "").casefold() or "finish" in str(c or "").casefold() for r in _table_rows(t) for c in r)]
            headers = _table_header(found["table"], header_tables)
            header_map = {i: h.casefold() for i, h in enumerate(headers)}
            answer_values = []
            def pick(*terms: str) -> Optional[str]:
                for i, value in enumerate(row):
                    heading = header_map.get(i, "")
                    if value and any(term in heading for term in terms):
                        return value.strip()
                return None
            if "width and height" in question.casefold():
                width, height = pick("width"), pick("height")
                if width and height:
                    answer_values = [f"{width} wide x {height} high"]
            elif "width" in question.casefold():
                value = pick("width")
                if value: answer_values = [value]
            elif "height" in question.casefold():
                value = pick("height")
                if value: answer_values = [value]
            elif "hardware group" in question.casefold():
                value = pick("hardware")
                if value: answer_values = [value]
            elif "fire rating" in question.casefold():
                value = pick("fire rating")
                if value: answer_values = [value]
            elif "frame material" in question.casefold():
                value = pick("frame material")
                if value: answer_values = ["HM (hollow metal)" if value.strip().upper() == "HM" else value]
            elif "door material" in question.casefold():
                value = pick("door material", "door matl")
                if value: answer_values = [value]
            elif "door type" in question.casefold():
                value = pick("door type")
                if value:
                    answer_values = [value]
                    description = _door_type_description(value, pages)
                    if description:
                        answer_values = [f"{value} ({description['text'].casefold()})"]
                        evidence.append({"page": description["page"], "bbox": description["bbox"],
                                         "text": description["text"], "source": document.get("source", {}).get("path")})
            elif "floor finish" in question.casefold():
                value = pick("floor finish")
                if value: answer_values = [value]
            elif "ceiling finish" in question.casefold():
                value = pick("ceiling finish")
                if value: answer_values = [value]
            elif "wall finish" in question.casefold():
                direction = next((d for d in ("north", "south", "east", "west") if d in question.casefold()), "")
                value = next((row[i].strip() for i, heading in header_map.items()
                              if direction and direction in heading and row[i]), None)
                if value: answer_values = [value]
            if answer_values:
                answer = answer_values[0]
                # Include a compact text definition when the schedule value is
                # a finish/type code and the sheet contains its legend entry.
                if re.fullmatch(r"[A-Z]\d(?:\s*&\s*[A-Z]\d)*", answer):
                    definitions = []
                    for code in re.findall(r"[A-Z]\d", answer):
                        for line in all_text.splitlines():
                            if re.match(rf"\s*{re.escape(code)}\s+", line, re.I):
                                definition = re.sub(rf"^\s*{re.escape(code)}\s*", "", line, flags=re.I).strip()
                                definition = re.sub(r"\s+Sports Complex$", "", definition, flags=re.I)
                                definitions.append(definition)
                                break
                    if definitions:
                        answer = f"{answer} - {' and '.join(definitions)}"
                evidence.append({
                    "page": found["table"].get("page", 1), "table_bbox": found["table"].get("bbox"),
                    "row": found["index"], "row_text": row,
                    "headers": headers, "source": document.get("source", {}).get("path"),
                })
                return _attach_pdf_page_sizes({"status": "answered", "answer": answer, "evidence": evidence}, pages)

    # Window schedule dimensions: this schedule is a compact 5-column table.
    if "window schedule" in question.casefold() and "width" in question.casefold() and "height" in question.casefold():
        candidates = [
            t for t in tables
            if not _is_border_dump_table(t)
            and any(len(row) >= 5 and str(row[0] or "").strip().isalpha() for row in _table_rows(t))
        ]
        if candidates:
            table = min(candidates, key=lambda t: len(_table_rows(t)))
            row = _table_rows(table)[0]
            if len(row) >= 5:
                return _attach_pdf_page_sizes({"status": "answered", "answer": f"{row[3]} wide x {row[4]} high", "evidence": [{
                    "page": 1, "table_bbox": table.get("bbox"), "row": 0,
                    "row_text": row, "source": document.get("source", {}).get("path"),
                }]}, pages)

    schedule_agg = _answer_pdf_schedule_aggregate(question, document)
    if schedule_agg is not None:
        return schedule_agg

    text_lookup = _answer_pdf_text_lookup(question, document)
    if text_lookup is not None:
        return text_lookup

    if enable_visual_fallback:
        visual = _answer_pdf_visual_fallback(
            question,
            document,
            render_dir=render_dir,
            model_path=model_path,
            geometry_model_path=geometry_model_path,
            cubicasa_repo_path=cubicasa_repo_path,
            cubicasa_weights_path=cubicasa_weights_path,
            object_detector_model=object_detector_model,
            query_plan=query_plan,
        )
        if visual is not None:
            return visual

    # For table or symbol counts, do not convert page-wide text mentions into
    # a count; the count may belong to a different view or schedule.
    return {"status": "not_found", "answer": None,
            "reason": "No unambiguous schedule row was found; this parser does not count drawn symbols.",
            "evidence": []}
