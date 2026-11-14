"""Parse raster technical drawings and answer grounded OCR or floor-plan questions."""

from __future__ import annotations

from pathlib import Path
import html as _html
import math
import os
import re
import struct
import sys
import threading
import unicodedata
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from floorplan_model import CATEGORY_NAMES, MODEL_VERSION, SharedFloorplanCounter, require_torch
from floorplan_geometry_model import (
    GEOMETRY_MODEL_VERSION,
    ICON_CLASSES,
    ICON_COUNT_CLASS,
    ROOM_CLASSES,
    ROOM_COUNT_CLASS,
    FloorplanGeometryNet,
    FurukawaHourglass,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_NON_FLOORPLAN_TYPES = {"elevation", "section", "detail", "site_plan", "map", "other"}
_COUNT_MODEL_CACHE: Dict[str, Tuple[Any, Any, int]] = {}
_GEOMETRY_MODEL_CACHE: Dict[str, Tuple[Any, Any, Any]] = {}
_CUBICASA_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any, Any]] = {}
_CUBICASA_PREDICTION_CACHE: Dict[Tuple[str, str, str, int, int], Dict[str, Any]] = {}
_GEOMETRIC_ANALYSIS_CACHE: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
_OBJECT_DETECTOR_CACHE: Dict[str, Any] = {}
_CUBICASA_DOWNLOAD_LOCK = threading.Lock()
_CUBICASA_DOWNLOAD_ATTEMPTED = False
_CUBICASA_GDRIVE_FILE_ID = "1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK"
_RAPID_OCR = None
_DIMENSION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:[.,]\d+)?\s*(?:mm|cm|ft|feet|inches|inch|in|m)(?![A-Za-z])",
    re.IGNORECASE,
)


def _read_setting(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val and val.strip():
        return val.strip()
    try:
        from llm_reasoner import get_env_setting
        return get_env_setting(name)
    except Exception:
        return None


def _auto_download_cubicasa_weights(target_path: Path) -> Optional[Path]:
    """Download the official CubiCasa5K checkpoint on first use when network is available."""
    global _CUBICASA_DOWNLOAD_ATTEMPTED
    with _CUBICASA_DOWNLOAD_LOCK:
        if target_path.is_file() and target_path.stat().st_size >= 100_000_000:
            return target_path
        if _CUBICASA_DOWNLOAD_ATTEMPTED:
            return None
        _CUBICASA_DOWNLOAD_ATTEMPTED = True

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            part_path = target_path.with_suffix(".pkl.part")
            url = (
                f"https://drive.usercontent.google.com/download"
                f"?id={_CUBICASA_GDRIVE_FILE_ID}&export=download&confirm=t"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                ctype = str(resp.headers.get("Content-Type") or "").lower()
                first_chunk = resp.read(65536)
                if "text/html" in ctype or first_chunk.lstrip().startswith((b"<!DOCTYPE", b"<html", b"<HTML")):
                    page_html = (first_chunk + resp.read(262144)).decode("utf-8", errors="ignore")
                    action_m = re.search(r'<form[^>]+action="([^"]+)"', page_html)
                    inputs = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page_html))
                    if not action_m:
                        return None
                    confirm_url = _html.unescape(action_m.group(1))
                    if inputs:
                        sep = "&" if "?" in confirm_url else "?"
                        confirm_url = confirm_url + sep + urllib.parse.urlencode(inputs)
                    req2 = urllib.request.Request(confirm_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req2, timeout=15.0) as resp2:
                        head = resp2.read(65536)
                        if not head.startswith((b"\x80", b"PK\x03\x04")):
                            return None
                        with part_path.open("wb") as out:
                            out.write(head)
                            while True:
                                buf = resp2.read(1024 * 1024)
                                if not buf:
                                    break
                                out.write(buf)
                else:
                    if not first_chunk.startswith((b"\x80", b"PK\x03\x04")):
                        return None
                    with part_path.open("wb") as out:
                        out.write(first_chunk)
                        while True:
                            buf = resp.read(1024 * 1024)
                            if not buf:
                                break
                            out.write(buf)

            if part_path.is_file() and part_path.stat().st_size >= 100_000_000:
                os.replace(part_path, target_path)
                return target_path
            if part_path.exists():
                part_path.unlink(missing_ok=True)
        except Exception:
            try:
                target_path.with_suffix(".pkl.part").unlink(missing_ok=True)
            except Exception:
                pass
        return None


def resolve_cubicasa_weights(
    weights_path: str | Path | None = None,
    *,
    auto_download: bool = True,
) -> Optional[Path]:
    """Locate CubiCasa5K weights across explicit, env, local data/, or cache paths."""
    if weights_path is not None:
        explicit = Path(weights_path).expanduser()
        return explicit.resolve() if explicit.is_file() else None

    env_weights = _read_setting("STOA_CUBICASA_WEIGHTS")
    if env_weights:
        env_p = Path(env_weights).expanduser()
        if env_p.is_file():
            return env_p.resolve()

    here = Path(__file__).resolve().parent
    candidates = [
        here / "data" / "model_best_val_loss_var.pkl",
        Path.cwd() / "data" / "model_best_val_loss_var.pkl",
        here.parent / "data" / "model_best_val_loss_var.pkl",
        Path.home() / "Stoa" / "data" / "model_best_val_loss_var.pkl",
        Path.home() / "data" / "model_best_val_loss_var.pkl",
        Path.home() / ".cache" / "stoa" / "model_best_val_loss_var.pkl",
    ]
    for cand in candidates:
        try:
            if cand.is_file() and cand.stat().st_size >= 100_000_000:
                return cand.resolve()
        except OSError:
            continue

    allow_dl = (_read_setting("STOA_AUTO_DOWNLOAD_CUBICASA") or "1").strip().lower() not in {"0", "false", "no"}
    if auto_download and allow_dl:
        return _auto_download_cubicasa_weights(here / "data" / "model_best_val_loss_var.pkl")
    return None


def default_cubicasa_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Return optional CubiCasa5K repo path and resolved checkpoint path."""
    here = Path(__file__).resolve().parent
    env_repo = _read_setting("STOA_CUBICASA_REPO")
    repo: Optional[Path] = None
    for cand in ([Path(env_repo).expanduser()] if env_repo else []) + [here / "data" / "CubiCasa5k"]:
        if (cand / "floortrans" / "models" / "hg_furukawa_original.py").is_file():
            repo = cand.resolve()
            break
    weights = resolve_cubicasa_weights()
    return repo, weights


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



def _load_cubicasa_model(repo_path: str | Path | None, weights_path: str | Path):
    """Load the 44-channel CubiCasa5K model (using built-in FurukawaHourglass when repo is absent)."""
    torch = require_torch()
    weights = Path(weights_path).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"CubiCasa5K pretrained checkpoint not found: {weights}")
    repo = Path(repo_path).expanduser().resolve() if repo_path else None
    repo_key = str(repo) if (repo and (repo / "floortrans" / "models" / "hg_furukawa_original.py").is_file()) else "__builtin__"
    key = (repo_key, str(weights))
    if key in _CUBICASA_MODEL_CACHE:
        return _CUBICASA_MODEL_CACHE[key]

    if repo_key != "__builtin__" and repo is not None:
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        from floortrans.models.hg_furukawa_original import hg_furukawa_original
        model = hg_furukawa_original(44)
    else:
        model = FurukawaHourglass(44)

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


def _cubicasa_prediction(image_path: str | Path, repo_path: str | Path | None, weights_path: str | Path):
    """Run official 44-channel CubiCasa inference and return localized classes."""
    torch = require_torch()
    import numpy as np
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("CubiCasa inference requires Pillow: pip install pillow") from exc

    image_file = Path(image_path).expanduser().resolve()
    repo_key = str(Path(repo_path).expanduser().resolve()) if repo_path else "__builtin__"
    weights = str(Path(weights_path).expanduser().resolve())
    stat = image_file.stat()
    key = (str(image_file), repo_key, weights, stat.st_size, stat.st_mtime_ns)
    if key in _CUBICASA_PREDICTION_CACHE:
        return _CUBICASA_PREDICTION_CACHE[key]

    model, device, _checkpoint = _load_cubicasa_model(repo_path, weights)
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

    def regions(probabilities, names, min_area, allowed_names=None, excluded_names=None):
        labels = probabilities.argmax(axis=0)
        result = []
        for class_id, name in enumerate(names):
            if class_id == 0:
                continue
            if allowed_names is not None and name not in allowed_names:
                continue
            if excluded_names is not None and name in excluded_names:
                continue
            mask = labels == class_id
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

    min_room_area = max(36, int(round(0.0035 * resized_size[0] * resized_size[1])))
    prediction = {
        "rooms": regions(room_prob, room_classes, min_room_area, excluded_names={"Outdoor", "Wall", "Railing"}),
        "outdoor_spaces": regions(room_prob, room_classes, min_room_area, allowed_names={"Outdoor"}),
        "icons": regions(icon_prob, icon_classes, 3),
        "model": "CubiCasa5K model_best_val_loss_var",
    }
    # Keep the parsed box summary rather than segmentation tensors so many
    # question rows for the same drawing reuse one inference with modest memory.
    if len(_CUBICASA_PREDICTION_CACHE) >= 1024:
        _CUBICASA_PREDICTION_CACHE.pop(next(iter(_CUBICASA_PREDICTION_CACHE)))
    _CUBICASA_PREDICTION_CACHE[key] = prediction
    return prediction


def analyze_floorplan_geometry(image_path: str | Path) -> Dict[str, Any]:
    """Extract wall topology, orthogonal alignment, and enclosed room regions purely from geometry.

    Small isolated connected components (text glyphs, dimension numbers, stipples) are removed
    before wall-footprint and distance-transform watershed analysis so text never influences
    room segmentation or floor-plan classification.
    """
    image_file = Path(image_path).expanduser().resolve()
    stat = image_file.stat()
    cache_key = (str(image_file), stat.st_size, stat.st_mtime_ns)
    if cache_key in _GEOMETRIC_ANALYSIS_CACHE:
        return _GEOMETRIC_ANALYSIS_CACHE[cache_key]

    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError:
        return {
            "is_floor_plan": False,
            "is_schedule_grid": False,
            "is_photo": False,
            "rooms": [],
            "outdoor_spaces": [],
            "icons": [],
            "thick_wall_ratio": 0.0,
            "ortho_energy": 0.0,
            "model": "geometric_wall_watershed",
        }

    with Image.open(image_file) as src_img:
        rgb0 = np.asarray(ImageOps.exif_transpose(src_img).convert("RGB"))
    h0, w0 = rgb0.shape[:2]
    scale = min(1.0, 800.0 / float(max(h0, w0, 1)))
    if scale < 1.0:
        w = max(64, int(round(w0 * scale)))
        h = max(64, int(round(h0 * scale)))
        img = cv2.resize(rgb0, (w, h), interpolation=cv2.INTER_AREA)
    else:
        img = rgb0
        h, w = h0, w0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]

    light_ratio = float((gray >= 200).mean())
    mid_photo_ratio = float(((gray >= 55) & (gray < 195) & (sat >= 55)).mean())

    # Dark low-saturation ink (walls/partitions); excludes bright colored site contours & fills
    ink_all = (((gray < 165) & (sat < 85)) | (gray < 95)).astype(np.uint8) * 255

    # Strip full-page sheet border lines along outer 3% margins so sheet borders do not trap flood-fill
    m_x = max(4, int(round(0.03 * w)))
    m_y = max(4, int(round(0.03 * h)))
    work_ink = ink_all.copy()
    for y0_b, y1_b in ((0, m_y), (h - m_y, h)):
        row_cov = (work_ink[y0_b:y1_b, :] > 0).mean(axis=1)
        for ry in np.where(row_cov > 0.55)[0]:
            yi = y0_b + int(ry)
            work_ink[max(0, yi - 2):min(h, yi + 3), :] = 0
    for x0_b, x1_b in ((0, m_x), (w - m_x, w)):
        col_cov = (work_ink[:, x0_b:x1_b] > 0).mean(axis=0)
        for rx in np.where(col_cov > 0.55)[0]:
            xi = x0_b + int(rx)
            work_ink[:, max(0, xi - 2):min(w, xi + 3)] = 0

    # Erase isolated text characters & small symbols (< 3.8% of minor image dimension)
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(work_ink, connectivity=8)
    min_span = max(14, int(round(0.038 * min(h, w))))
    min_area = max(24, int(round(0.00022 * h * w)))
    keep = np.zeros(num_cc, dtype=bool)
    if num_cc > 1:
        spans = np.maximum(cc_stats[1:, cv2.CC_STAT_WIDTH], cc_stats[1:, cv2.CC_STAT_HEIGHT])
        areas = cc_stats[1:, cv2.CC_STAT_AREA]
        keep[1:] = (spans >= min_span) & (areas >= min_area)
    clean_wall = (keep[cc_labels].astype(np.uint8)) * 255

    thick_px = max(3, int(round(0.005 * min(h, w))))
    thick_wall = cv2.morphologyEx(
        clean_wall, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (thick_px, thick_px))
    )
    thick_wall_ratio = float((thick_wall > 0).sum()) / float(max((clean_wall > 0).sum(), 1))

    gx = cv2.Sobel(clean_wall, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clean_wall, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    strong = mag > 40.0
    if np.any(strong):
        ang180 = (np.degrees(np.arctan2(gy[strong], gx[strong]))) % 180.0
        bins180 = np.histogram(ang180, bins=36, range=(0.0, 180.0), weights=mag[strong])[0]
        total_w = float(bins180.sum()) + 1e-6
        ortho_pairs = bins180[:18] + bins180[18:]
        best_idx = int(np.argmax(ortho_pairs))
        idx_set = [(best_idx - 1) % 18, best_idx, (best_idx + 1) % 18]
        axis_a = sum(float(bins180[i]) for i in idx_set) / total_w
        axis_b = sum(float(bins180[i + 18]) for i in idx_set) / total_w
        ortho_energy = axis_a + axis_b
        ortho_balance = min(axis_a, axis_b) / max(axis_a, axis_b, 1e-6)
    else:
        ortho_energy = 0.0
        ortho_balance = 0.0

    ys, xs = np.where(clean_wall > 0)
    rooms: List[Dict[str, Any]] = []
    footprint_ratio = 0.0
    if len(xs) >= 90:
        bx0, bx1 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
        by0, by1 = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
        b_span = max(48, min(bx1 - bx0, by1 - by0))

        outer_gap = max(11, int(round(0.085 * b_span))) | 1
        envelope = cv2.morphologyEx(
            clean_wall, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_gap, outer_gap))
        )
        padded = np.zeros((h + 4, w + 4), dtype=np.uint8)
        padded[2:h + 2, 2:w + 2] = envelope
        ff = padded.copy()
        cv2.floodFill(ff, np.zeros((h + 6, w + 6), dtype=np.uint8), (0, 0), 128)
        footprint = ~(ff[2:h + 2, 2:w + 2] == 128)
        footprint_area = int(footprint.sum())
        footprint_ratio = footprint_area / float(h * w)

        if footprint_area >= 0.015 * h * w:
            interior_free = footprint & (clean_wall == 0)
            dist = cv2.distanceTransform((interior_free.astype(np.uint8)) * 255, cv2.DIST_L2, 5)
            sigma = max(1.5, 0.006 * b_span)
            dist_smooth = cv2.GaussianBlur(dist, (0, 0), sigmaX=sigma, sigmaY=sigma)

            min_peak_dist = max(11, int(round(0.075 * b_span))) | 1
            dilated = cv2.dilate(
                dist_smooth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_peak_dist, min_peak_dist))
            )
            min_room_radius = max(4.5, 0.022 * b_span)
            local_max = (dist_smooth >= dilated - 1e-4) & (dist_smooth >= min_room_radius) & interior_free

            n_pk, pk_labels = cv2.connectedComponents(local_max.astype(np.uint8), connectivity=8)
            ws_markers = np.where(pk_labels > 0, pk_labels + 1, 0).astype(np.int32)
            ws_markers[~interior_free] = 1

            inv_dist = np.clip(
                255.0 - (dist_smooth / max(float(dist_smooth.max()), 1.0)) * 255.0, 0, 255
            ).astype(np.uint8)
            cv2.watershed(cv2.cvtColor(inv_dist, cv2.COLOR_GRAY2BGR), ws_markers)

            basin_peak = np.zeros(n_pk + 2, dtype=np.float32)
            valid_b = ws_markers >= 2
            if np.any(valid_b):
                np.maximum.at(basin_peak, ws_markers[valid_b], dist[valid_b])

            ridge_ys, ridge_xs = np.where((ws_markers == -1) & interior_free)
            pair_max_saddle: Dict[Tuple[int, int], float] = {}
            if len(ridge_xs) > 0:
                valid_r = (ridge_ys >= 1) & (ridge_ys < h - 1) & (ridge_xs >= 1) & (ridge_xs < w - 1)
                ry, rx = ridge_ys[valid_r], ridge_xs[valid_r]
                rd = dist[ry, rx]
                for l1_arr, l2_arr in (
                    (ws_markers[ry, rx - 1], ws_markers[ry, rx + 1]),
                    (ws_markers[ry - 1, rx], ws_markers[ry + 1, rx]),
                ):
                    diff_m = (l1_arr >= 2) & (l2_arr >= 2) & (l1_arr != l2_arr)
                    for a, b, dval in zip(l1_arr[diff_m].tolist(), l2_arr[diff_m].tolist(), rd[diff_m].tolist()):
                        key_ab = (a, b) if a < b else (b, a)
                        if dval > pair_max_saddle.get(key_ab, 0.0):
                            pair_max_saddle[key_ab] = dval

            parent = list(range(n_pk + 2))

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for (a, b), saddle in pair_max_saddle.items():
                min_pk = min(float(basin_peak[a]), float(basin_peak[b]))
                if min_pk > 0 and (saddle / min_pk) >= 0.76:
                    ra, rb = _find(a), _find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

            remap = np.array([_find(i) if i >= 2 else 0 for i in range(n_pk + 2)], dtype=np.int32)
            labels_map = remap[np.where(ws_markers >= 2, ws_markers, 0)]

            min_room_px = max(85, int(round(0.008 * footprint_area)))
            inv_scale = 1.0 / scale
            if labels_map.max() >= 2:
                counts = np.bincount(labels_map.ravel())
                for rid in np.where(counts >= min_room_px)[0]:
                    if rid < 2:
                        continue
                    reg = labels_map == rid
                    area_px = int(counts[rid])
                    rys, rxs = np.where(reg)
                    rx0, ry0, rx1, ry1 = int(rxs.min()), int(rys.min()), int(rxs.max()), int(rys.max())
                    rw, rh = rx1 - rx0 + 1, ry1 - ry0 + 1
                    aspect = max(rw, rh) / max(min(rw, rh), 1)
                    fill_frac = area_px / float(max(rw * rh, 1))
                    if aspect > 8.5 or fill_frac < 0.20:
                        continue
                    rooms.append({
                        "class": "Room",
                        "confidence": 0.70,
                        "bbox": [
                            max(0, int(round(rx0 * inv_scale))),
                            max(0, int(round(ry0 * inv_scale))),
                            min(w0, int(round((rx1 + 1) * inv_scale))),
                            min(h0, int(round((ry1 + 1) * inv_scale))),
                        ],
                        "area_px": int(round(area_px * inv_scale * inv_scale)),
                        "_h_px": rh,
                    })

    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, w // 5), 1))
    vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, h // 5)))
    h_grid = cv2.morphologyEx(clean_wall, cv2.MORPH_OPEN, horiz_k) > 0
    v_grid = cv2.morphologyEx(clean_wall, cv2.MORPH_OPEN, vert_k) > 0
    grid_dominance = float((h_grid | v_grid).sum()) / float(max((clean_wall > 0).sum(), 1))

    h_cv = (
        float(np.std([r["_h_px"] for r in rooms]) / max(np.mean([r["_h_px"] for r in rooms]), 1.0))
        if len(rooms) >= 5
        else 1.0
    )
    for r in rooms:
        r.pop("_h_px", None)

    is_schedule_grid = (
        (grid_dominance > 0.65 and thick_wall_ratio < 0.03)
        or (grid_dominance > 0.72 and thick_wall_ratio < 0.18 and (h_cv < 0.28 or len(rooms) > 24))
    )
    is_photo = (light_ratio < 0.30 or mid_photo_ratio > 0.32)

    is_floor_plan = (
        not is_photo
        and not is_schedule_grid
        and light_ratio >= 0.38
        and ortho_energy >= 0.42
        and ortho_balance >= 0.15
        and 0.015 <= footprint_ratio <= 0.985
        and (
            (len(rooms) >= 2 and (thick_wall_ratio >= 0.06 or ortho_energy >= 0.62))
            or (len(rooms) >= 1 and thick_wall_ratio >= 0.30 and ortho_energy >= 0.65)
        )
    )

    result = {
        "is_floor_plan": bool(is_floor_plan),
        "is_schedule_grid": bool(is_schedule_grid),
        "is_photo": bool(is_photo),
        "rooms": rooms,
        "outdoor_spaces": [],
        "icons": [],
        "thick_wall_ratio": round(thick_wall_ratio, 4),
        "ortho_energy": round(ortho_energy, 4),
        "model": "geometric_wall_watershed",
    }
    if len(_GEOMETRIC_ANALYSIS_CACHE) >= 512:
        _GEOMETRIC_ANALYSIS_CACHE.pop(next(iter(_GEOMETRIC_ANALYSIS_CACHE)))
    _GEOMETRIC_ANALYSIS_CACHE[cache_key] = result
    return result


def is_likely_floorplan_by_structure(
    image_path: str | Path,
    cubicasa_repo_path: str | Path | None = None,
    cubicasa_weights_path: str | Path | None = None,
) -> bool:
    """Verify floor-plan layout via CubiCasa5K room+opening segmentation or pure wall geometry."""
    geo = analyze_floorplan_geometry(image_path)
    if geo.get("is_photo") or geo.get("is_schedule_grid"):
        return False

    weights = resolve_cubicasa_weights(cubicasa_weights_path)
    if weights is not None:
        try:
            pred = _cubicasa_prediction(image_path, cubicasa_repo_path, weights)
            strong_rooms = [r for r in pred.get("rooms", []) if float(r.get("confidence", 0.0)) >= 0.50]
            strong_openings = [
                ic for ic in pred.get("icons", [])
                if ic.get("class") in {"Door", "Window"} and float(ic.get("confidence", 0.0)) >= 0.50
            ]
            if len(strong_rooms) >= 2 and len(strong_openings) >= 2:
                # Guard against thin-line facade elevations having 2 false rooms when thick walls are absent
                if len(strong_rooms) >= 3 or geo.get("is_floor_plan") or float(geo.get("thick_wall_ratio", 0.0)) >= 0.05:
                    return True
            return False
        except Exception:
            pass

    return bool(geo.get("is_floor_plan"))


def enrich_image_specialist_evidence(
    document: Dict[str, Any],
    *,
    cubicasa_repo_path: str | Path | None = None,
    cubicasa_weights_path: str | Path | None = None,
    geometry_model_path: str | Path | None = None,
    include_rotated_ocr: bool = False,
) -> Dict[str, Any]:
    """Attach cached floor-plan segmentation and room-label OCR to a parsed image."""
    source = document.get("source", {}).get("path")
    drawing_type = document.get("drawing_type")
    if not source or drawing_type in _NON_FLOORPLAN_TYPES:
        return document

    if "floorplan_specialist" not in document:
        resolved_weights = resolve_cubicasa_weights(cubicasa_weights_path)
        if resolved_weights is not None:
            try:
                document["floorplan_specialist"] = _cubicasa_prediction(
                    source, cubicasa_repo_path, resolved_weights
                )
            except Exception as exc:
                document["floorplan_specialist_error"] = f"{type(exc).__name__}: {exc}"
        elif geometry_model_path is not None:
            try:
                geom = _geometry_prediction(source, geometry_model_path)
                excluded = {"background", "outdoor", "wall", "railing"}
                document["floorplan_specialist"] = {
                    "rooms": [
                        r for r in geom.get("rooms", [])
                        if str(r.get("class", "")).casefold() not in excluded
                    ],
                    "outdoor_spaces": [
                        r for r in geom.get("rooms", [])
                        if str(r.get("class", "")).casefold() == "outdoor"
                    ],
                    "icons": geom.get("icons", []),
                    "model": "floorplan_geometry_checkpoint",
                }
            except Exception as exc:
                document["floorplan_specialist_error"] = f"{type(exc).__name__}: {exc}"
        else:
            geo = analyze_floorplan_geometry(source)
            if geo.get("rooms"):
                document["floorplan_specialist"] = {
                    "rooms": geo["rooms"],
                    "outdoor_spaces": [],
                    "icons": [],
                    "model": geo["model"],
                }

    if "ocr_room_labels" not in document:
        try:
            from room_label_qa import SUPPORTED as ROOM_OCR_SUPPORTED, _hits, read_labels
            width = int(document.get("width") or 0)
            height = int(document.get("height") or 0)
            if width > 0 and height > 0:
                if include_rotated_ocr or not document.get("text"):
                    rot_labels = read_labels(source)
                else:
                    rot_labels = {
                        0: [
                            {
                                "text": str(w.get("text", "")),
                                "conf": float(w.get("confidence", 0.0)),
                                "bbox": list(w.get("bbox", [0, 0, 0, 0])),
                            }
                            for w in document.get("text", [])
                            if w.get("text") and w.get("bbox")
                        ]
                    }
                extracted: List[Dict[str, Any]] = []
                for cat in ROOM_OCR_SUPPORTED:
                    if cat == "storeys":
                        continue
                    for hit in _hits(rot_labels, cat, (width, height)):
                        extracted.append({
                            "category": cat,
                            "label": hit["label"],
                            "ocr_text": hit["ocr_text"],
                            "confidence": hit["conf"],
                            "bbox": hit["bbox"],
                        })
                document["ocr_room_labels"] = extracted
        except Exception:
            document["ocr_room_labels"] = []

    return document


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
    query_plan: Dict[str, Any] | None = None,
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
    if object_detector_model and drawing_type in _NON_FLOORPLAN_TYPES:
        detected = _answer_raster_object_count(question, document, object_detector_model)
        if detected is not None:
            return detected

    if drawing_type in _NON_FLOORPLAN_TYPES:
        return {"status": "unsupported", "answer": None,
                "reason": "This drawing type has OCR support, but no configured detector supports this question.",
                "evidence": []}

    resolved_cubicasa_weights = (
        resolve_cubicasa_weights(cubicasa_weights_path)
        if model_path is None and geometry_model_path is None
        else resolve_cubicasa_weights(cubicasa_weights_path, auto_download=False)
    )

    from train_floorplan_model import question_category

    category = question_category(question)
    folded_q = _fold_text(question)
    asks_count = any(term in folded_q for term in (
        "how many", "number of", "count", "quantity of", "combien", "wie viele",
        "anzahl", "kuinka monta", "montako", "hvor mange", "mitu",
    ))
    asks_general_rooms = (
        (query_plan is not None and query_plan.get("intent") == "count" and query_plan.get("target") in {"rooms", "spaces"})
        or (
            category is None
            and asks_count
            and re.search(r"\b(?:rooms?|spaces?|huonetta|huoneita|raume|pieces|locali)\b", folded_q) is not None
        )
    )

    if asks_general_rooms:
        scope = str((query_plan or {}).get("scope") or "").casefold()
        whole_drawing_scopes = (
            "this floor plan", "current floor plan", "this plan", "whole plan", "entire plan",
            "this drawing", "whole drawing", "entire drawing", "current drawing", "this floor",
        )
        sub_area_cues = ("apartment ", "wohnung ", "appartement ", "unit ", "level ", "storey ", "floor 1", "floor 2")
        if (scope and not any(term in scope for term in whole_drawing_scopes)) or (
            not scope and any(cue in folded_q for cue in sub_area_cues)
        ):
            return {
                "status": "unsupported", "answer": None,
                "reason": "The image parser cannot isolate a room count to the requested sub-area or level yet.",
                "evidence": [evidence],
            }
        if resolved_cubicasa_weights is not None:
            geometry = _cubicasa_prediction(source, cubicasa_repo_path, resolved_cubicasa_weights)
            document["floorplan_specialist"] = geometry
            regions = geometry["rooms"]
            model_name = geometry["model"]
            method = "cubicasa5k_indoor_room_segmentation_components"
        elif geometry_model_path is not None:
            geometry = _geometry_prediction(source, geometry_model_path)
            document["floorplan_specialist"] = geometry
            excluded = {"background", "outdoor", "wall", "railing"}
            regions = [item for item in geometry["rooms"]
                       if str(item.get("class", "")).casefold() not in excluded]
            model_name = "floorplan_geometry_checkpoint"
            method = "shared_geometry_indoor_room_components"
        else:
            geo = analyze_floorplan_geometry(source)
            regions = geo["rooms"]
            document["floorplan_specialist"] = {
                "rooms": regions,
                "outdoor_spaces": [],
                "icons": [],
                "model": geo["model"],
            }
            model_name = geo["model"]
            method = "geometric_wall_topology_segmentation"

        if not regions:
            return {
                "status": "not_found", "answer": None,
                "reason": "The segmentation model returned no indoor room regions; this does not establish a count of zero.",
                "method": method, "model": model_name, "review_recommended": True,
                "evidence": [evidence],
            }
        region_evidence = [
            {"source": source, "page": 1, "bbox": item["bbox"],
             "coordinate_system": "image pixels; origin at top-left",
             "class": item.get("class"), "confidence": item.get("confidence")}
            for item in regions
        ]
        by_class: Dict[str, int] = {}
        for item in regions:
            cls_name = str(item.get("class") or "Room")
            by_class[cls_name] = by_class.get(cls_name, 0) + 1
        breakdown = ", ".join(f"{cnt} {cls}" for cls, cnt in by_class.items())
        return {
            "status": "answered", "value": len(regions), "answer": len(regions), "unit": "count",
            "answer_text": f"{len(regions)} indoor rooms ({breakdown})",
            "reason": f"Summed all {len(regions)} detected indoor rooms across types: {breakdown}.",
            "method": method, "model": model_name, "review_recommended": True,
            "evidence": region_evidence,
        }

    def _room_label_fallback(target_category: Optional[str]) -> Optional[Dict[str, Any]]:
        if not target_category or not source:
            return None
        from room_label_qa import SUPPORTED as ROOM_OCR_SUPPORTED, answer_room_question
        if target_category not in ROOM_OCR_SUPPORTED:
            return None
        room_res = answer_room_question(question, source, target_category)
        if room_res.get("status") != "answered":
            return None
        raw_ev = room_res.get("evidence") or []
        label_items = raw_ev[0].get("labels", []) if raw_ev and isinstance(raw_ev[0], dict) else []
        formatted_ev = [
            {
                "source": source,
                "page": 1,
                "bbox": item["bbox"],
                "coordinate_system": "image pixels; origin at top-left",
                "label": item.get("label"),
                "ocr_text": item.get("ocr_text"),
                "confidence": item.get("conf"),
            }
            for item in label_items
            if item.get("bbox")
        ] or [evidence]
        count_val = int(room_res["value"])
        return {
            "status": "answered",
            "value": count_val,
            "answer": count_val,
            "unit": "count",
            "answer_text": f"{count_val} {target_category.replace('_', ' ')}",
            "method": room_res.get("method", "room_label_ocr"),
            "review_recommended": bool(room_res.get("review_recommended")) or drawing_type != "floor_plan",
            "evidence": formatted_ev,
        }

    if resolved_cubicasa_weights is not None and category:
        if category == "showers_bathtubs" and any(
                term in question.casefold() for term in ("shower", "suihku")):
            return {"status": "not_found", "answer": None,
                    "reason": "CubiCasa5K has a bathtub class but no shower class, so this question cannot be answered reliably.",
                    "evidence": [evidence]}
        class_for_category = {
            "windows": ("icons", "Window"), "doors": ("icons", "Door"),
            "toilet_bowls": ("icons", "Toilet"), "sinks": ("icons", "Sink"),
            "showers_bathtubs": ("icons", "Bathtub"),
            "bedrooms": ("rooms", "Bedroom"), "kitchens": ("rooms", "Kitchen"),
            "bathrooms": ("rooms", "Bath"),
        }
        target = class_for_category.get(category)
        if target is None:
            ocr_room_result = _room_label_fallback(category)
            if ocr_room_result is not None:
                return ocr_room_result
            return {"status": "not_found", "answer": None,
                    "reason": f"CubiCasa5K has no direct segmentation class for {category}.",
                    "evidence": [evidence]}
        geometry = _cubicasa_prediction(source, cubicasa_repo_path, resolved_cubicasa_weights)
        document["floorplan_specialist"] = geometry
        regions = [item for item in geometry[target[0]] if item["class"] == target[1]]
        region_evidence = [
            {"source": source, "page": 1, "bbox": item["bbox"],
             "coordinate_system": "image pixels; origin at top-left",
             "class": item["class"], "confidence": item["confidence"]}
            for item in regions
        ] or [evidence]
        answer_label = "bathtubs" if category == "showers_bathtubs" else category.replace("_", " ")
        return {
            "status": "answered", "value": len(regions), "answer": len(regions), "unit": "count",
            "answer_text": f"{len(regions)} {answer_label}",
            "method": "cubicasa5k_pretrained_segmentation_components",
            "model": geometry["model"], "review_recommended": True,
            "evidence": region_evidence,
        }

    if geometry_model_path is not None and category:
        path = source
        if not path:
            raise ValueError("Parsed image is missing its source path")
        geometry = _geometry_prediction(path, geometry_model_path)
        document["floorplan_specialist"] = geometry
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
            ] or [evidence]
            auxiliary = geometry["auxiliary_counts"].get(category)
            return {
                "status": "answered",
                "value": count,
                "answer": count,
                "unit": "count",
                "answer_text": f"{count} {category.replace('_', ' ')}",
                "method": method,
                "auxiliary_count_estimate": auxiliary,
                "review_recommended": (auxiliary is not None and auxiliary != count) or drawing_type != "floor_plan",
                "evidence": region_evidence,
            }

        ocr_room_result = _room_label_fallback(category)
        if ocr_room_result is not None:
            return ocr_room_result

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
        est_count = geometry["auxiliary_counts"][category]
        return {
            "status": "answered",
            "value": est_count,
            "answer": est_count,
            "unit": "count",
            "answer_text": f"{est_count} {category.replace('_', ' ')}",
            "method": "weakly_supervised_count_head",
            "review_recommended": True,
            "evidence": [evidence],
        }

    if model_path is not None and category:
        image_path = source
        if not image_path:
            raise ValueError("Parsed image is missing its source path")
        count, model_info = _count_prediction(image_path, model_path, category)
        label = category.replace("_", " ")
        return {
            "status": "answered",
            "value": count,
            "answer": count,
            "unit": "count",
            "answer_text": f"{count} {label}",
            "method": "shared_cnn_count_head",
            "model": model_info,
            "review_recommended": True,
            "evidence": [evidence],
        }

    if category in {"wc_rooms", "saunas", "balconies", "storeys"}:
        ocr_room_result = _room_label_fallback(category)
        if ocr_room_result is not None:
            return ocr_room_result

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
