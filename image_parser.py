"""Read raster drawing metadata and return verifiable image evidence.

The parser deliberately does not infer symbol counts from image dimensions or
OCR.  A vision/OCR backend can enrich the returned OCR records later; until
then, visual count questions are reported as unsupported instead of guessed.
"""

from __future__ import annotations

from pathlib import Path
import re
import struct
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _png_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    return None


def _jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Read JPEG SOF dimensions without requiring a raster imaging library."""
    if not data.startswith(b"\xff\xd8"):
        return None
    index = 2
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                   0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while index < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            continue
        if index + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[index:index + 2])[0]
        if segment_length < 2 or index + segment_length > len(data):
            break
        if marker in sof_markers and segment_length >= 7:
            height, width = struct.unpack(">HH", data[index + 3:index + 7])
            return width, height
        index += segment_length
    return None


def _dimensions(path: Path) -> Tuple[int, int]:
    # Reading the whole file keeps the metadata reader dependency-free and the
    # dataset images are small enough for this approach.
    data = path.read_bytes()
    dims = _png_dimensions(data) or _jpeg_dimensions(data)
    if dims:
        return dims

    # Pillow provides dimensions for TIFF/BMP and uncommon JPEG variants.
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot read dimensions for {path.suffix} without Pillow. "
            "Install it with: pip install pillow"
        ) from exc
    try:
        with Image.open(path) as image:
            return image.size
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {path}") from exc


def _drawing_type(path: Path) -> str:
    if any(part.casefold() in {"cubicasa5k", "swiss_dwellings"} for part in path.parts):
        return "floor_plan"
    return "raster_drawing"


def _ocr(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Run Tesseract OCR when both pytesseract and its executable are present."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return "unavailable", []
    try:
        data = pytesseract.image_to_data(Image.open(path), output_type=pytesseract.Output.DICT)
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}", []

    words = []
    for i, value in enumerate(data.get("text", [])):
        text = str(value).strip()
        if not text:
            continue
        left, top = int(data["left"][i]), int(data["top"][i])
        width, height = int(data["width"][i]), int(data["height"][i])
        words.append({
            "text": text,
            "confidence": float(data["conf"][i]),
            "bbox": [left, top, left + width, top + height],
            "page": 1,
        })
    return "available", words


def parse_image(file_path: str | Path, *, run_ocr: bool = True) -> Dict[str, Any]:
    """Extract raster metadata and optional OCR words with pixel coordinates."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix or '(none)'}")
    width, height = _dimensions(path)
    ocr_status, words = _ocr(path) if run_ocr else ("skipped", [])
    return {
        "source": {"path": str(path), "format": path.suffix.lstrip(".").upper()},
        "drawing_type": _drawing_type(path),
        "width": width,
        "height": height,
        "ocr_status": ocr_status,
        "text": words,
        "evidence_region": {
            "page": 1,
            "bbox": [0, 0, width, height],
            "coordinate_system": "image pixels; origin at top-left",
        },
    }


def answer_image_question(question: str, document: Dict[str, Any]) -> Dict[str, Any]:
    """Answer literal OCR lookups; defer visual reasoning without a vision backend.

    The result schema matches the PDF answerer: ``answered`` results carry
    pixel-coordinate evidence, while unsupported questions state why.
    """
    source = document.get("source", {}).get("path")
    evidence = {
        "source": source,
        "page": 1,
        "bbox": document.get("evidence_region", {}).get("bbox"),
        "coordinate_system": "image pixels; origin at top-left",
    }
    words = document.get("text", [])
    if words:
        # Only return OCR terms explicitly requested as text, never use OCR to
        # fabricate a symbol count or assert the meaning of a room abbreviation.
        match = re.search(r"(?:read|what(?:'s| is)?(?: the)?(?: text| label)?)\s*[\"']([^\"']+)[\"']", question, re.I)
        if match:
            wanted = match.group(1).casefold()
            found = [word for word in words if wanted in str(word.get("text", "")).casefold()]
            if found:
                return {
                    "status": "answered",
                    "answer": " ".join(word["text"] for word in found),
                    "evidence": [{**evidence, "bbox": found[0].get("bbox"), "ocr_words": found}],
                }

    return {
        "status": "unsupported",
        "answer": None,
        "reason": (
            "This image question requires visual symbol recognition or spatial reasoning. "
            "The current image parser only extracts metadata and optional OCR text."
        ),
        "evidence": [evidence],
    }
