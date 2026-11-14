"""Parse raster technical drawings and answer grounded OCR or floor-plan questions."""

from __future__ import annotations

from pathlib import Path
import math
import re
import struct
import sys
import unicodedata
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
_CUBICASA_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any, Any]] = {}
_OBJECT_DETECTOR_CACHE: Dict[str, Any] = {}
_RAPID_OCR = None
_DIMENSION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:[.,]\d+)?\s*(?:mm|cm|ft|feet|inches|inch|in|m)(?![A-Za-z])",
    re.IGNORECASE,
)


def ocr_image_array(image: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the shared RapidOCR engine on an RGB array and return pixel boxes."""
    global _RAPID_OCR
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
        if _RAPID_OCR is None:
            _RAPID_OCR = RapidOCR()
        result, _elapsed = _RAPID_OCR(np.asarray(image))
        words = []
        for detection in result or []:
            box, text, confidence = detection
            text = str(text).strip()
            if not text:
                continue
            points = np.asarray(box, dtype=float).reshape(-1, 2)
            x0, y0 = np.floor(points.min(axis=0)).astype(int)
            x1, y1 = np.ceil(points.max(axis=0)).astype(int)
            words.append({"text": text, "confidence": float(confidence),
                          "bbox": [int(x0), int(y0), int(x1), int(y1)]})
        return "available", words
    except ImportError:
        return "unavailable", []
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}", []


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
    """Read raster drawing text with RapidOCR, falling back to Tesseract."""
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            status, detected = ocr_image_array(image)
        if status == "unavailable":
            raise ImportError("rapidocr_onnxruntime is not installed")
        if status.startswith("failed:"):
            raise RuntimeError(status)
        return status, [{**word, "page": 1} for word in detected]
    except ImportError:
        # Keep a fallback for local environments that have Tesseract installed
        # but have not yet installed RapidOCR's ONNX runtime.
        try:
            from PIL import Image, ImageOps
            import pytesseract
        except ImportError:
            return "unavailable", []
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
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
                "text": text, "confidence": float(data["conf"][i]),
                "bbox": [left, top, left + width, top + height], "page": 1,
            })
        return "available", words
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}", []


def _ocr_lines(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group word detections into reading lines while retaining pixel boxes."""
    ordered = sorted(words, key=lambda word: (
        (word["bbox"][1] + word["bbox"][3]) / 2, word["bbox"][0]
    ))
    clusters: List[List[Dict[str, Any]]] = []
    for word in ordered:
        x0, y0, x1, y1 = word["bbox"]
        center = (y0 + y1) / 2
        height = max(1, y1 - y0)
        candidates = []
        for index, cluster in enumerate(clusters):
            centers = [(item["bbox"][1] + item["bbox"][3]) / 2 for item in cluster]
            heights = [max(1, item["bbox"][3] - item["bbox"][1]) for item in cluster]
            tolerance = max(5, 0.55 * max(height, sum(heights) / len(heights)))
            distance = abs(center - sum(centers) / len(centers))
            if distance <= tolerance:
                candidates.append((distance, index))
        if candidates:
            clusters[min(candidates)[1]].append(word)
        else:
            clusters.append([word])

    lines = []
    for cluster in clusters:
        cluster.sort(key=lambda word: word["bbox"][0])
        boxes = [word["bbox"] for word in cluster]
        text = " ".join(word["text"] for word in cluster)
        lines.append({
            "text": text,
            "bbox": [min(box[0] for box in boxes), min(box[1] for box in boxes),
                     max(box[2] for box in boxes), max(box[3] for box in boxes)],
            "confidence": round(sum(float(word["confidence"]) for word in cluster) / len(cluster), 4),
        })
    return sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))


def _fold_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text)).casefold().replace(chr(248), "o").replace(chr(230), "ae")
    return "".join(character for character in value if not unicodedata.combining(character))


def _drawing_metadata(lines: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    patterns = {
        "scale": re.compile(r"\b(?:scale|massstab|echelle|mittakaava|mootkava|maalestok)\b|(?<!\d)1\s*[:/]\s*\d{2,4}", re.I),
        "drawing_number": re.compile(r"\b(?:drawing|sheet|plan)\s*(?:number|no\.?|#)|\b(?:blattnummer|plannummer|lehenr|piirustusnumero|tegningsnr|zeichnungsnummer|numero de dessin)\b", re.I),
        "date": re.compile(r"\b(?:date|datum|drawn|paivays)\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", re.I),
        "title": re.compile(r"\b(?:drawing\s+title|project\s+name|sheet\s+title|title|titel|titre|projektin nimi|piirustuksen nimi)\b", re.I),
    }
    return {
        kind: [line for line in lines if pattern.search(_fold_text(line["text"]))]
        for kind, pattern in patterns.items()
    }


def parse_image(file_path: str | Path, *, run_ocr: bool = True) -> Dict[str, Any]:
    """Extract raster metadata and optional OCR words with pixel coordinates."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix or '(none)'}")
    width, height = _dimensions(path)
    ocr_status, words = _ocr(path) if run_ocr else ("skipped", [])
    lines = _ocr_lines(words)
    measurements = [
        {"text": match.group(0), "bbox": line["bbox"], "confidence": line["confidence"], "page": 1}
        for line in lines for match in _DIMENSION_PATTERN.finditer(line["text"])
    ]
    return {
        "source": {"path": str(path), "format": path.suffix.lstrip(".").upper()},
        "drawing_type": _drawing_type(path),
        "width": width,
        "height": height,
        "ocr_status": ocr_status,
        "text": words,
        "lines": lines,
        "measurements": measurements,
        "drawing_metadata": _drawing_metadata(lines),
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



def _load_cubicasa_model(repo_path: str | Path, weights_path: str | Path):
    """Load the official CubiCasa5K model and its published checkpoint."""
    torch = require_torch()
    repo = Path(repo_path).expanduser().resolve()
    weights = Path(weights_path).expanduser().resolve()
    key = (str(repo), str(weights))
    if key in _CUBICASA_MODEL_CACHE:
        return _CUBICASA_MODEL_CACHE[key]
    if not (repo / "floortrans" / "models" / "hg_furukawa_original.py").is_file():
        raise FileNotFoundError(f"CubiCasa5K checkout is missing floortrans/models: {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"CubiCasa5K pretrained checkpoint not found: {weights}")
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    # Instantiate the official architecture directly. The released CubiCasa
    # checkpoint contains every parameter, so inference does not need to load
    # the separate ImageNet/pose initialization weights used before training.
    from floortrans.models.hg_furukawa_original import hg_furukawa_original
    model = hg_furukawa_original(44)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(weights, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Expected the official CubiCasa checkpoint with a 'model_state' entry")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    loaded = (model, device, checkpoint)
    _CUBICASA_MODEL_CACHE[key] = loaded
    return loaded


def _cubicasa_prediction(image_path: str | Path, repo_path: str | Path, weights_path: str | Path):
    """Run official 44-channel CubiCasa inference and return localized classes."""
    torch = require_torch()
    import numpy as np
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("CubiCasa inference requires Pillow: pip install pillow") from exc

    model, device, _checkpoint = _load_cubicasa_model(repo_path, weights_path)
    with Image.open(image_path) as source_image:
        image = ImageOps.exif_transpose(source_image).convert("RGB")
        original_width, original_height = image.size
        # Match the upstream F1_scaled model input scale and pad to the network's
        # 64-pixel stride so the decoder output aligns with the input.
        scale = min(1.0, 512.0 / max(original_width, original_height))
        resized_size = (max(1, round(original_width * scale)), max(1, round(original_height * scale)))
        image = image.resize(resized_size, Image.Resampling.LANCZOS) if resized_size != image.size else image
        input_width = ((resized_size[0] + 63) // 64) * 64
        input_height = ((resized_size[1] + 63) // 64) * 64
        canvas = Image.new("RGB", (input_width, input_height), "white")
        canvas.paste(image, (0, 0))
        tensor = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        tensor = tensor.reshape(input_height, input_width, 3).permute(2, 0, 1).float().div(127.5).sub(1.0)

    with torch.no_grad():
        outputs = model(tensor.unsqueeze(0).to(device))[0]
        room_prob = torch.softmax(outputs[21:33], dim=0).cpu().numpy()
        icon_prob = torch.softmax(outputs[33:44], dim=0).cpu().numpy()

    room_classes = ("Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom",
                    "Bath", "Hallway", "Railing", "Storage", "Garage", "Other rooms")
    icon_classes = ("Empty", "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
                    "Sink", "Sauna bench", "Fireplace", "Bathtub", "Chimney")
    scale_x, scale_y = original_width / resized_size[0], original_height / resized_size[1]

    def regions(probabilities, names, min_area, room=False):
        labels = probabilities.argmax(axis=0)
        result = []
        for class_id, name in enumerate(names):
            if class_id == 0:
                continue
            mask = labels == class_id
            if room:
                # The upstream room labels include the whole wall/room polygon;
                # exclude the common structural classes from room-object results.
                if name in {"Outdoor", "Wall", "Railing"}:
                    continue
            for component in _connected_components(mask, min_area):
                x0, y0, x1, y1 = (component[k] for k in ("x0", "y0", "x1", "y1"))
                confidence = float(probabilities[class_id, y0:y1, x0:x1].mean())
                result.append({
                    "class": name,
                    "bbox": [max(0, int(x0 * scale_x)), max(0, int(y0 * scale_y)),
                             min(original_width, int(math.ceil(x1 * scale_x))),
                             min(original_height, int(math.ceil(y1 * scale_y)))],
                    "confidence": round(confidence, 4),
                    "area_px": int(round(component["area"] * scale_x * scale_y)),
                })
        return result

    return {
        "rooms": regions(room_prob, room_classes, 12, room=True),
        "icons": regions(icon_prob, icon_classes, 3),
        "model": "CubiCasa5K model_best_val_loss_var",
    }


def _answer_raster_text_question(question: str, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Answer only text/metadata questions grounded in OCR line evidence."""
    source = document.get("source", {}).get("path")
    lines = document.get("lines", [])
    metadata = document.get("drawing_metadata", {})
    normalized_question = _fold_text(question)

    def result_for(line: Dict[str, Any], method: str, answer: Optional[str] = None):
        return {
            "status": "answered", "answer": answer if answer is not None else line["text"],
            "method": method,
            "evidence": [{"source": source, "page": 1, "bbox": line["bbox"],
                          "coordinate_system": "image pixels; origin at top-left",
                          "ocr_text": line["text"], "ocr_confidence": line["confidence"]}],
        }

    quoted = re.search(r"[\"'](.+?)[\"']|[\u2018\u201c](.+?)[\u2019\u201d]", question)
    if quoted:
        target = _fold_text(quoted.group(1) or quoted.group(2)).strip()
        for line in lines:
            if target and target in _fold_text(line["text"]):
                return result_for(line, "raster_ocr_exact_text")

    metadata_questions = {
        "scale": ("scale", "massstab", "echelle", "mittakaava", "mootkava", "maalestok"),
        "drawing_number": ("drawing number", "drawing no", "sheet number", "sheet no", "plan number", "plan no", "lehe number", "blattnummer", "piirustusnumero", "tegningsnr", "zeichnungsnummer", "numero de dessin"),
        "date": ("date", "datum", "drawn on", "paivays"),
        "title": ("drawing title", "sheet title", "project name", "title of the drawing", "titre du dessin", "projektin nimi", "piirustuksen nimi"),
    }
    for kind, terms in metadata_questions.items():
        if any(term in normalized_question for term in terms):
            matches = metadata.get(kind, [])
            if matches:
                line = matches[0]
                return result_for(line, f"raster_ocr_{kind}")

    dimension_terms = ("width", "height", "length", "diameter", "breite", "hohe", "longueur", "largeur", "hauteur", "leveys", "korkeus", "pituus", "breedte", "hoogte")
    asks_dimension = any(term in normalized_question for term in (
        "dimension", "dimensions", *dimension_terms, "measurement", "size"
    ))
    if asks_dimension:
        candidates = document.get("measurements", [])
        if candidates:
            dimension_lines = [line for line in lines if _DIMENSION_PATTERN.search(line["text"])]
            question_terms = [term for term in dimension_terms if term in normalized_question]
            if question_terms:
                # Only answer a named dimension when OCR links its label and
                # value on the same line; otherwise abstain instead of guessing.
                selected = [line for line in dimension_lines
                            if any(term in _fold_text(line["text"]) for term in question_terms)][:5]
                if not selected:
                    return {"status": "not_found", "answer": None,
                            "reason": "No printed value linked to that named dimension was found.",
                            "evidence": []}
            else:
                selected = dimension_lines[:5]
            return {
                "status": "answered", "answer": [line["text"] for line in selected],
                "method": "raster_ocr_printed_dimensions",
                "evidence": [{"source": source, "page": 1, "bbox": line["bbox"],
                              "coordinate_system": "image pixels; origin at top-left",
                              "ocr_text": line["text"], "ocr_confidence": line["confidence"]}
                             for line in selected],
            }

    if any(term in normalized_question for term in ("what text", "read the text", "what label", "read the label")):
        return {"status": "not_found", "answer": None,
                "reason": "No matching quoted label was found in the OCR text.", "evidence": []}
    return None


def _answer_raster_object_count(
    question: str, document: Dict[str, Any], model_id: str | Path
) -> Optional[Dict[str, Any]]:
    """Try an optional zero-shot box detector for countable drawing symbols."""
    normalized = _fold_text(question)
    if not any(term in normalized for term in ("how many", "count", "number of", "quantity of")):
        return None
    labels = (
        ("windows", ("window", "windows", "fenster", "fenetre")),
        ("doors", ("door", "doors", "porte", "tuer")),
        ("stairs", ("stair", "stairs", "staircase", "staircases")),
        ("columns", ("column", "columns", "pillar", "pillars")),
        ("beams", ("beam", "beams", "girder", "girders")),
        ("trees", ("tree", "trees")),
        ("cars", ("car", "cars", "vehicle", "vehicles")),
    )
    match = next(((label, terms) for label, terms in labels
                  if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms)), None)
    if match is None:
        return None
    label, _terms = match
    model_key = str(model_id)
    try:
        detector = _OBJECT_DETECTOR_CACHE.get(model_key)
        if detector is None:
            from transformers import pipeline
            device = 0 if require_torch().cuda.is_available() else -1
            detector = pipeline("zero-shot-object-detection", model=model_key, device=device)
            _OBJECT_DETECTOR_CACHE[model_key] = detector
        from PIL import Image
        source = document.get("source", {}).get("path")
        if not source:
            return {"status": "not_found", "answer": None,
                    "reason": "The parsed drawing has no source image path.", "evidence": []}
        with Image.open(source) as stream:
            image = stream.convert("RGB")
        detections = detector(image, candidate_labels=[label])
    except Exception as exc:
        return {"status": "not_found", "answer": None,
                "reason": f"Object detector could not run: {type(exc).__name__}: {exc}", "evidence": []}

    accepted = [item for item in detections if float(item.get("score", 0.0)) >= 0.35]
    if not accepted:
        return {"status": "not_found", "answer": None,
                "reason": f"The detector found no {label} boxes above its confidence threshold; this does not establish a count of zero.",
                "method": "grounding_dino_zero_shot", "model": model_key,
                "review_recommended": True, "evidence": []}
    evidence = []
    for item in accepted:
        box = item.get("box", {})
        evidence.append({
            "source": source, "page": 1,
            "bbox": [int(box.get("xmin", 0)), int(box.get("ymin", 0)),
                     int(box.get("xmax", 0)), int(box.get("ymax", 0))],
            "coordinate_system": "image pixels; origin at top-left",
            "class": item.get("label", label), "confidence": float(item.get("score", 0.0)),
        })
    return {"status": "answered", "answer": len(accepted),
            "answer_text": f"{len(accepted)} detected {label}",
            "method": "grounding_dino_zero_shot", "model": model_key,
            "review_recommended": True, "evidence": evidence}

def answer_image_question(
    question: str,
    document: Dict[str, Any],
    *,
    model_path: str | Path | None = None,
    geometry_model_path: str | Path | None = None,
    cubicasa_repo_path: str | Path | None = None,
    cubicasa_weights_path: str | Path | None = None,
    object_detector_model: str | Path | None = None,
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

    text_answer = _answer_raster_text_question(question, document)
    if text_answer is not None:
        return text_answer

    drawing_type = document.get("drawing_type")
    if object_detector_model and drawing_type not in {"floor_plan", "uncertain", None}:
        detected = _answer_raster_object_count(question, document, object_detector_model)
        if detected is not None:
            return detected

    if drawing_type not in {"floor_plan", "uncertain", None}:
        return {"status": "unsupported", "answer": None,
                "reason": "This drawing type has OCR support, but no configured detector supports this question.",
                "evidence": []}

    if (cubicasa_repo_path is not None and cubicasa_weights_path is not None
            and document.get("drawing_type") == "floor_plan"):
        from train_floorplan_model import question_category

        category = question_category(question)
        if category:
            if category == "showers_bathtubs" and any(
                    term in question.casefold() for term in ("shower", "suihku")):
                return {"status": "not_found", "answer": None,
                        "reason": "CubiCasa5K has a bathtub class but no shower class, so this question cannot be answered reliably.",
                        "evidence": [evidence]}
            geometry = _cubicasa_prediction(source, cubicasa_repo_path, cubicasa_weights_path)
            class_for_category = {
                "windows": ("icons", "Window"), "doors": ("icons", "Door"),
                "toilet_bowls": ("icons", "Toilet"), "sinks": ("icons", "Sink"),
                "showers_bathtubs": ("icons", "Bathtub"),
                "bedrooms": ("rooms", "Bedroom"), "kitchens": ("rooms", "Kitchen"),
                "bathrooms": ("rooms", "Bath"),
            }
            target = class_for_category.get(category)
            if target is None:
                return {"status": "not_found", "answer": None,
                        "reason": f"CubiCasa5K has no direct segmentation class for {category}.",
                        "evidence": [evidence]}
            regions = [item for item in geometry[target[0]] if item["class"] == target[1]]
            region_evidence = [
                {"source": source, "page": 1, "bbox": item["bbox"],
                 "coordinate_system": "image pixels; origin at top-left",
                 "class": item["class"], "confidence": item["confidence"]}
                for item in regions
            ] or [evidence]
            answer_label = "bathtubs" if category == "showers_bathtubs" else category.replace("_", " ")
            return {
                "status": "answered", "answer": len(regions),
                "answer_text": f"{len(regions)} {answer_label}",
                "method": "cubicasa5k_pretrained_segmentation_components",
                "model": geometry["model"], "review_recommended": True,
                "evidence": region_evidence,
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
            "This raster parser can answer OCR text, drawing metadata, and printed-dimension "
            "lookups. Symbol counts and geometry-derived quantities for this drawing class "
            "need a dedicated vision parser; no reliable matching evidence was found."
        ),
        "evidence": [evidence],
    }
