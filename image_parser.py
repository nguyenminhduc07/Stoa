"""Read raster drawings and answer floor-plan count questions with a CNN."""

from __future__ import annotations

from pathlib import Path
import math
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

from floorplan_model import CATEGORY_NAMES, MODEL_VERSION, SharedFloorplanCounter, require_torch
from floorplan_geometry_model import (
    GEOMETRY_MODEL_VERSION,
    ICON_CLASSES,
    ICON_COUNT_CLASS,
    ROOM_CLASSES,
    ROOM_COUNT_CLASS,
    FloorplanGeometryNet,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_COUNT_MODEL_CACHE: Dict[str, Tuple[Any, Any, int]] = {}
_GEOMETRY_MODEL_CACHE: Dict[str, Tuple[Any, Any, Any]] = {}


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


def _load_count_model(checkpoint_path: str | Path):
    torch = require_torch()
    path = Path(checkpoint_path).resolve()
    key = str(path)
    if key in _COUNT_MODEL_CACHE:
        return _COUNT_MODEL_CACHE[key]
    if not path.is_file():
        raise FileNotFoundError(f"Floor-plan model checkpoint not found: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Unsupported floor-plan model version: {checkpoint.get('model_version')}")
    categories = checkpoint.get("categories", [])
    if tuple(categories) != CATEGORY_NAMES:
        raise ValueError("Checkpoint count categories do not match this parser version")
    model = SharedFloorplanCounter(categories).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    cached = (model, checkpoint, device)
    _COUNT_MODEL_CACHE[key] = cached
    return cached


def _count_prediction(image_path: str | Path, checkpoint_path: str | Path, category: str) -> Tuple[int, Dict[str, Any]]:
    torch = require_torch()
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("CNN image inference requires Pillow: pip install pillow") from exc

    model, checkpoint, device = _load_count_model(checkpoint_path)
    image_size = int(checkpoint["image_size"])
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        canvas.paste(image, ((image_size - image.width) // 2, (image_size - image.height) // 2))
        pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        tensor = pixels.reshape(image_size, image_size, 3).permute(2, 0, 1).float().div(127.5).sub(1.0)
    with torch.no_grad():
        prediction = model(tensor.unsqueeze(0).to(device))[category].item()
    count = max(0, int(round(math.expm1(max(0.0, prediction)))))
    mae = checkpoint.get("validation_mae_counts", {}).get(category)
    return count, {
        "model_version": checkpoint["model_version"],
        "category": category,
        "validation_mae_count": mae,
    }


def _load_geometry_model(checkpoint_path: str | Path):
    torch = require_torch()
    path = Path(checkpoint_path).resolve()
    key = str(path)
    if key in _GEOMETRY_MODEL_CACHE:
        return _GEOMETRY_MODEL_CACHE[key]
    if not path.is_file():
        raise FileNotFoundError(f"Geometry model checkpoint not found: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_version") != GEOMETRY_MODEL_VERSION:
        raise ValueError(f"Unsupported geometry model version: {checkpoint.get('model_version')}")
    if tuple(checkpoint.get("rooms", [])) != ROOM_CLASSES or tuple(checkpoint.get("icons", [])) != ICON_CLASSES:
        raise ValueError("Geometry checkpoint class mapping does not match this parser")
    if tuple(checkpoint.get("count_categories", [])) != CATEGORY_NAMES:
        raise ValueError("Geometry checkpoint count heads do not match this parser")
    model = FloorplanGeometryNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    cached = (model, checkpoint, device)
    _GEOMETRY_MODEL_CACHE[key] = cached
    return cached


def _connected_components(mask, minimum_area: int) -> List[Dict[str, int]]:
    """Return 8-connected component bounds, using OpenCV when available."""
    import numpy as np

    binary = np.asarray(mask, dtype=np.uint8)
    if not binary.any():
        return []
    try:
        import cv2
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        return [
            {"x0": int(x), "y0": int(y), "x1": int(x + width), "y1": int(y + height), "area": int(area)}
            for x, y, width, height, area in stats[1:count]
            if int(area) >= minimum_area
        ]
    except ImportError:
        height, width = binary.shape
        visited = np.zeros(binary.shape, dtype=np.bool_)
        components = []
        for y, x in np.argwhere(binary):
            y, x = int(y), int(x)
            if visited[y, x]:
                continue
            visited[y, x] = True
            stack = [(y, x)]
            x0 = x1 = x
            y0 = y1 = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                x0, x1 = min(x0, cx), max(x1, cx)
                y0, y1 = min(y0, cy), max(y1, cy)
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if area >= minimum_area:
                components.append({"x0": x0, "y0": y0, "x1": x1 + 1, "y1": y1 + 1, "area": area})
        return components


def _geometry_prediction(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    confidence_threshold: float = 0.45,
) -> Dict[str, Any]:
    torch = require_torch()
    import numpy as np
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Geometry inference requires Pillow: pip install pillow") from exc

    model, checkpoint, device = _load_geometry_model(checkpoint_path)
    size = int(checkpoint["image_size"])
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        original_width, original_height = image.size
        scale = min(size / original_width, size / original_height)
        resized_size = (max(1, round(original_width * scale)), max(1, round(original_height * scale)))
        image = image.resize(resized_size, Image.Resampling.LANCZOS)
        offset = ((size - resized_size[0]) // 2, (size - resized_size[1]) // 2)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(image, offset)
        data = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        tensor = data.reshape(size, size, 3).permute(2, 0, 1).float().div(127.5).sub(1.0)

    with torch.no_grad():
        outputs = model(tensor.unsqueeze(0).to(device))
        room_prob = torch.softmax(outputs["rooms"][0], dim=0).cpu().numpy()
        icon_prob = torch.softmax(outputs["icons"][0], dim=0).cpu().numpy()
        count_log = outputs["count_log1p"][0].cpu().numpy()

    x0, y0 = offset
    resized_width, resized_height = resized_size
    room_prob = room_prob[:, y0:y0 + resized_height, x0:x0 + resized_width]
    icon_prob = icon_prob[:, y0:y0 + resized_height, x0:x0 + resized_width]
    scale_x, scale_y = original_width / resized_width, original_height / resized_height

    def extract_regions(probabilities, class_names, minimum_area, excluded=(0,)):
        labels = probabilities.argmax(axis=0)
        confidence = probabilities.max(axis=0)
        result = []
        for class_id, class_name in enumerate(class_names):
            if class_id in excluded:
                continue
            class_mask = (labels == class_id) & (probabilities[class_id] >= confidence_threshold)
            for component in _connected_components(class_mask, minimum_area):
                bounds = component
                bbox = [
                    max(0, int(bounds["x0"] * scale_x)),
                    max(0, int(bounds["y0"] * scale_y)),
                    min(original_width, int(math.ceil(bounds["x1"] * scale_x))),
                    min(original_height, int(math.ceil(bounds["y1"] * scale_y))),
                ]
                region_confidence = float(probabilities[class_id,
                    bounds["y0"]:bounds["y1"], bounds["x0"]:bounds["x1"]].mean())
                result.append({
                    "class": class_name,
                    "bbox": bbox,
                    "area_px": int(round(component["area"] * scale_x * scale_y)),
                    "confidence": round(region_confidence, 4),
                })
        return result

    # Small icon regions are retained; room geometry uses a larger area floor
    # to suppress isolated pixel noise.
    icons = extract_regions(icon_prob, ICON_CLASSES, minimum_area=3)
    rooms = extract_regions(room_prob, ROOM_CLASSES, minimum_area=12)
    auxiliary_counts = {
        category: max(0, int(round(math.expm1(max(0.0, float(count_log[index]))))))
        for index, category in enumerate(CATEGORY_NAMES)
    }
    return {
        "source": str(Path(image_path).resolve()),
        "width": original_width,
        "height": original_height,
        "icons": icons,
        "rooms": rooms,
        "auxiliary_counts": auxiliary_counts,
        "count_mae": checkpoint.get("metrics", {}).get("qa_count_mae", {}),
        "confidence_threshold": confidence_threshold,
    }


def answer_image_question(
    question: str,
    document: Dict[str, Any],
    *,
    model_path: str | Path | None = None,
    geometry_model_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Answer OCR lookups or learned image-level count questions.

    Count predictions cite the image bounds. The QA-supervised model does not
    predict object locations, so callers should present the result as a count
    estimate and make the full drawing available for review.
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

    if geometry_model_path is not None and document.get("drawing_type") == "floor_plan":
        from train_floorplan_model import question_category

        category = question_category(question)
        if category:
            path = source
            if not path:
                raise ValueError("Parsed image is missing its source path")
            geometry = _geometry_prediction(path, geometry_model_path)
            icon_class = ICON_COUNT_CLASS.get(category)
            room_class = ROOM_COUNT_CLASS.get(category)
            if icon_class:
                regions = [region for region in geometry["icons"] if region["class"] == icon_class]
                method = "geometry_icon_mask_components"
            elif room_class:
                regions = [region for region in geometry["rooms"] if region["class"] == room_class]
                method = "geometry_room_mask_components"
            else:
                regions = None

            if regions is not None:
                count = len(regions)
                region_evidence = [
                    {"source": path, "page": 1, "bbox": region["bbox"],
                     "coordinate_system": "image pixels; origin at top-left",
                     "class": region["class"], "confidence": region["confidence"]}
                    for region in regions
                ]
                if not region_evidence:
                    region_evidence = [evidence]
                auxiliary = geometry["auxiliary_counts"].get(category)
                return {
                    "status": "answered",
                    "answer": count,
                    "answer_text": f"{count} {category.replace('_', ' ')}",
                    "method": method,
                    "auxiliary_count_estimate": auxiliary,
                    "review_recommended": auxiliary is not None and auxiliary != count,
                    "evidence": region_evidence,
                }

            # Some QA categories (for example storeys and shower fixtures) do
            # not have matching classes in the CubiCasa masks. Keep their
            # learned image-level estimate separate from geometric detections.
            if geometry["count_mae"].get(category) is None:
                return {
                    "status": "not_found",
                    "answer": None,
                    "reason": (
                        f"The geometry taxonomy has no direct class for {category}, "
                        "and this checkpoint has no QA supervision for its auxiliary count head."
                    ),
                    "evidence": [evidence],
                }
            return {
                "status": "answered",
                "answer": geometry["auxiliary_counts"][category],
                "answer_text": f"{geometry['auxiliary_counts'][category]} {category.replace('_', ' ')}",
                "method": "weakly_supervised_count_head",
                "review_recommended": True,
                "evidence": [evidence],
            }

    if model_path is not None and document.get("drawing_type") == "floor_plan":
        from train_floorplan_model import question_category

        category = question_category(question)
        if category:
            image_path = source
            if not image_path:
                raise ValueError("Parsed image is missing its source path")
            count, model_info = _count_prediction(image_path, model_path, category)
            label = category.replace("_", " ")
            return {
                "status": "answered",
                "answer": count,
                "answer_text": f"{count} {label}",
                "method": "shared_cnn_count_head",
                "model": model_info,
                "review_recommended": True,
                "evidence": [evidence],
            }

    return {
        "status": "unsupported",
        "answer": None,
        "reason": (
            "This image question needs visual reasoning. Supply a trained floor-plan "
            "count-model checkpoint for supported count questions; the current QA-trained "
            "model estimates image-level totals and does not localize individual objects."
        ),
        "evidence": [evidence],
    }
