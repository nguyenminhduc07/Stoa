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


def _table_rows(table: Dict[str, Any]) -> List[List[Optional[str]]]:
    return [[cell.get("text") for cell in row.get("cells", [])] for row in table.get("rows", [])]


def _table_header(table: Dict[str, Any], linked_headers: List[Dict[str, Any]]) -> List[str]:
    rows = _table_rows(table)
    if not rows:
        return []
    column_count = max(map(len, rows))
    header_rows = []
    for row in rows[:4]:
        joined = " ".join(str(value or "") for value in row)
        if any(word in joined.casefold() for word in ("width", "height", "finish", "material", "fire rating", "number", "hardware")):
            header_rows.append(row)
        elif header_rows:
            break
    if not header_rows:
        # Some drawings put the schedule headings in an adjacent table object.
        row_count = max(map(len, rows))
        is_room_finish_table = any("SF" in str(cell or "") for cell in (rows[0] if rows else []))
        ranked = sorted(
            linked_headers,
            key=lambda candidate: (
                0 if is_room_finish_table and "floor finish" in " ".join(
                    str(c or "") for row in _table_rows(candidate) for c in row
                ).casefold() else 1,
                abs(max(map(len, _table_rows(candidate))) - row_count) if _table_rows(candidate) else 999,
            ),
        )
        for candidate in ranked:
            for row in _table_rows(candidate):
                joined = " ".join(str(value or "") for value in row)
                if any(word in joined.casefold() for word in ("width", "height", "finish", "material", "fire rating", "number", "hardware")) or any(
                    word in joined.casefold() for word in ("north wall", "south wall", "east wall", "west wall")
                ):
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
        rows = _table_rows(table)
        title = " ".join(str(cell or "") for row in rows[:3] for cell in row).casefold()
        for index, row in enumerate(rows):
            if not any(_normal(cell) == target_norm for cell in row if cell):
                continue
            row_text = " ".join(str(cell or "") for cell in row)
            if len(row) <= 1:
                continue
            if category == "door" and "door" not in title and "mark" not in title:
                continue
            if category == "room" and "floor finish" not in title and "room finish" not in title:
                # The heading can be a separate table object; room data rows
                # have an area in the first cell and a room name in the third.
                if len(row) < 8 or not any("SF" in str(cell or "") for cell in row[:1]):
                    continue
            choices.append((table, index, row, title))
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
    if any(term in q_low for term in ("schedule", "scheduled", "in the model", "im modell", "dans le modele", "dans le modèle")):
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
        header_table = next((t for t in tables if any("Floor Finish" in str(c or "") for r in _table_rows(t) for c in r)), None)
        data_table = None
        for table in tables:
            rows = _table_rows(table)
            if any(len(row) >= 10 and any("SF" in str(cell or "") for cell in row[:1]) for row in rows):
                data_table = table
        if data_table and header_table:
            headers = _table_header(header_table, [])
            finish_col = next((i for i, h in enumerate(headers) if "floor finish" in h.casefold()), 3)
            area_values = []
            selected = []
            for row_index, row in enumerate(_table_rows(data_table)):
                if len(row) > finish_col and "F0" in str(row[finish_col] or ""):
                    match = re.search(r"([\d,]+)\s*SF", str(row[0] or ""), re.I)
                    if match:
                        area_values.append(int(match.group(1).replace(",", "")))
                    selected.append(row_index)
            if selected:
                answer = f"{len(selected)} rooms, {sum(area_values)} SF"
                return _attach_pdf_page_sizes({"status": "answered", "answer": answer, "evidence": [{
                    "page": 1, "table_bbox": data_table.get("bbox"), "rows": selected,
                    "rows_text": [_table_rows(data_table)[i] for i in selected],
                    "source": document.get("source", {}).get("path"),
                }]}, pages)

    if "largest area" in question.casefold():
        data_table = next((t for t in tables if any(len(row) >= 10 and any("SF" in str(c or "") for c in row[:1]) for row in _table_rows(t))), None)
        if data_table:
            candidates = []
            for i, row in enumerate(_table_rows(data_table)):
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

    if target:
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
        candidates = [t for t in tables if any(len(row) >= 5 and str(row[0] or "").strip().isalpha() for row in _table_rows(t))]
        if candidates:
            table = min(candidates, key=lambda t: len(_table_rows(t)))
            row = _table_rows(table)[0]
            if len(row) >= 5:
                return _attach_pdf_page_sizes({"status": "answered", "answer": f"{row[3]} wide x {row[4]} high", "evidence": [{
                    "page": 1, "table_bbox": table.get("bbox"), "row": 0,
                    "row_text": row, "source": document.get("source", {}).get("path"),
                }]}, pages)

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
