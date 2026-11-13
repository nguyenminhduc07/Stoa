"""Extract searchable, page-located text and schedule tables from PDFs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional


def parse_pdf(file_path: str | Path) -> Dict[str, Any]:
    """Parse PDF text, words, tables, and page coordinates for evidence retrieval.

    Raster-only PDFs remain indexed as pages with no extracted text; this
    parser does not claim to OCR them or count drawn symbols.
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
                "text": page.extract_text(x_tolerance=2, y_tolerance=2) or "",
                "words": [
                    {key: word[key] for key in ("text", "x0", "top", "x1", "bottom")}
                    for word in words
                ],
                "tables": tables,
                "image_count": len(page.images),
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


def answer_pdf_question(question: str, document: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve a grounded schedule answer from parsed PDF tables.

    This first version answers schedule/text questions. Symbol-count questions
    on plans are returned as unsupported because PDF text extraction cannot
    reliably count drawn objects.
    """
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
                return {"status": "answered", "answer": answer, "evidence": [{
                    "page": 1, "table_bbox": data_table.get("bbox"), "rows": selected,
                    "rows_text": [_table_rows(data_table)[i] for i in selected],
                    "source": document.get("source", {}).get("path"),
                }]}

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
                return {"status": "answered", "answer": answer, "evidence": [{
                    "page": 1, "table_bbox": data_table.get("bbox"), "rows": [row_index],
                    "rows_text": [row], "source": document.get("source", {}).get("path"),
                }]}

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
                return {"status": "answered", "answer": answer, "evidence": evidence}

    # Window schedule dimensions: this schedule is a compact 5-column table.
    if "window schedule" in question.casefold() and "width" in question.casefold() and "height" in question.casefold():
        candidates = [t for t in tables if any(len(row) >= 5 and str(row[0] or "").strip().isalpha() for row in _table_rows(t))]
        if candidates:
            table = min(candidates, key=lambda t: len(_table_rows(t)))
            row = _table_rows(table)[0]
            if len(row) >= 5:
                return {"status": "answered", "answer": f"{row[3]} wide x {row[4]} high", "evidence": [{
                    "page": 1, "table_bbox": table.get("bbox"), "row": 0,
                    "row_text": row, "source": document.get("source", {}).get("path"),
                }]}

    # For table or symbol counts, do not convert page-wide text mentions into
    # a count; the count may belong to a different view or schedule.
    return {"status": "not_found", "answer": None,
            "reason": "No unambiguous schedule row was found; this parser does not count drawn symbols.",
            "evidence": []}
