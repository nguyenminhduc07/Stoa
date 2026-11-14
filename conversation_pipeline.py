"""Conversation-scoped file ingestion, parser caching, and grounded question routing.

Create one ConversationPipeline per chat. It holds only in-memory metadata and
parsed results; callers should discard it when that conversation ends.
"""

from __future__ import annotations

import hashlib
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_FORMAT_BY_EXTENSION = {
    ".ifc": "ifc", ".dxf": "dxf", ".dwg": "dwg", ".pdf": "pdf",
    **{extension: "image" for extension in _IMAGE_EXTENSIONS},
}
_FORMAT_PRIORITY = {"ifc": 0, "dxf": 1, "dwg": 1, "pdf": 2, "image": 3}
_AREA_TERMS = ("area", "surface", "superficie", "flache", "areal", "pinta-ala", "pindala")
_COUNT_TERMS = ("how many", "number of", "count", "combien", "nombre de", "wie viele", "anzahl",
                "quante", "quanti", "cuantos", "cuantas", "kuinka monta", "montako", "hvor mange", "mitu", "how much")


@dataclass
class FileEntry:
    file_id: str
    path: Path
    format: str
    size_bytes: int
    group_id: str
    drawing_type: Optional[str] = None
    parse_status: str = "not_parsed"
    parse_error: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = field(default=None, repr=False)
    ifc_index: Any = field(default=None, repr=False)

    def public(self, related_ids: Iterable[str] = ()) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "name": self.path.name,
            "path": str(self.path),
            "format": self.format,
            "drawing_type": self.drawing_type,
            "size_bytes": self.size_bytes,
            "group_id": self.group_id,
            "related_file_ids": list(related_ids),
            "parse_status": self.parse_status,
            "parse_error": self.parse_error,
        }


class ConversationPipeline:
    """Manage files and grounded answers for one conversation.

    Upload handlers should first save incoming files to a conversation-owned
    directory, then pass those paths to ``add_files``. Parsing is lazy and each
    document is parsed at most once until ``close`` is called.
    """

    def __init__(
        self,
        *,
        count_model: Optional[str | Path] = None,
        geometry_model: Optional[str | Path] = None,
        cubicasa_repo: Optional[str | Path] = None,
        cubicasa_weights: Optional[str | Path] = None,
    ) -> None:
        self.count_model = Path(count_model) if count_model else None
        self.geometry_model = Path(geometry_model) if geometry_model else None
        self.cubicasa_repo = Path(cubicasa_repo) if cubicasa_repo else None
        self.cubicasa_weights = Path(cubicasa_weights) if cubicasa_weights else None
        self.files: Dict[str, FileEntry] = {}
        self._groups: Dict[str, List[str]] = {}
        self._active_group: Optional[str] = None
        self._lock = threading.RLock()
        self._vision_classifier = None
        self.classification_warnings: Dict[str, str] = {}

    @staticmethod
    def _file_id(path: Path) -> str:
        stat = path.stat()
        identity = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8")
        return hashlib.sha256(identity).hexdigest()[:12]

    @staticmethod
    def _drawing_group(path: Path) -> str:
        return path.stem.casefold()

    @staticmethod
    def _detect_format(path: Path) -> str:
        """Use extension and lightweight file signatures; never use dataset source names."""
        suffix = path.suffix.casefold()
        try:
            with path.open("rb") as stream:
                header = stream.read(4096)
        except OSError:
            return "unsupported"
        if header.startswith(b"%PDF-"):
            return "pdf"
        if b"ISO-10303-21" in header[:1024]:
            return "ifc"
        if header.startswith(b"AC10"):
            return "dwg"
        if (header.startswith(b"\x89PNG\r\n\x1a\n") or header.startswith(b"\xff\xd8")
                or header.startswith(b"BM") or header.startswith((b"II*\x00", b"MM\x00*"))):
            return "image"
        return _FORMAT_BY_EXTENSION.get(suffix, "unsupported")

    def add_files(self, paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
        """Register uploaded paths and return their manifest entries.

        The caller owns upload copying and cleanup. Re-adding an unchanged path
        returns the existing ID and preserves its parsed cache.
        """
        added: List[str] = []
        for value in paths:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Uploaded file not found: {path}")
            file_id = self._file_id(path)
            if file_id in self.files:
                added.append(file_id)
                continue
            group_id = self._drawing_group(path)
            entry = FileEntry(
                file_id=file_id,
                path=path,
                format=self._detect_format(path),
                size_bytes=path.stat().st_size,
                group_id=group_id,
            )
            if entry.format == "image":
                entry.drawing_type = self._classify_image(path, file_id)
            self.files[file_id] = entry
            self._groups.setdefault(group_id, []).append(file_id)
            added.append(file_id)
        return [self._manifest_entry(file_id) for file_id in added]

    def _classify_image(self, path: Path, file_id: str) -> str:
        """Use the shared zero-shot labels when Transformers/model weights exist."""
        try:
            from PIL import Image
            from transformers import pipeline
            from file_classifier import DocumentRouter

            if self._vision_classifier is None:
                # CPU avoids sending CLIP through a possibly unsupported GPU
                # wheel (the user's V100 needs a build that includes sm_70).
                self._vision_classifier = pipeline(
                    task="zero-shot-image-classification", model=DocumentRouter.IMAGE_MODEL,
                    device=-1,
                )
            with Image.open(path) as source:
                image = source.convert("RGB")
            predictions = self._vision_classifier(image, candidate_labels=DocumentRouter.IMAGE_LABELS)
            if not predictions:
                self.classification_warnings[file_id] = "Classifier returned no candidate labels."
                return "uncertain"
            best = predictions[0]
            second = predictions[1]["score"] if len(predictions) > 1 else 0.0
            margin = float(best["score"]) - float(second)
            if best["score"] < 0.20 or margin < 0.04:
                self.classification_warnings[file_id] = (
                    f"Low classifier confidence: {best['label']}={best['score']:.3f}, "
                    f"runner-up={second:.3f}, margin={margin:.3f}."
                )
                return "uncertain"
            return DocumentRouter.LABEL_TO_TYPE.get(best["label"], "uncertain")
        except Exception as exc:
            self.classification_warnings[file_id] = f"{type(exc).__name__}: {exc}"
            return "uncertain"

    def _manifest_entry(self, file_id: str) -> Dict[str, Any]:
        entry = self.files[file_id]
        related = [item for item in self._groups.get(entry.group_id, []) if item != file_id]
        result = entry.public(related)
        warning = self.classification_warnings.get(file_id)
        if warning:
            result["classification_warning"] = warning
        return result

    def manifest(self) -> List[Dict[str, Any]]:
        """Return files currently registered in this conversation."""
        return [self._manifest_entry(file_id) for file_id in self.files]

    @staticmethod
    def _normalise(text: str) -> str:
        value = unicodedata.normalize("NFKD", text).casefold()
        return "".join(character for character in value if not unicodedata.combining(character))

    def _intent(self, question: str) -> str:
        text = self._normalise(question)
        if any(term in text for term in _AREA_TERMS) or "m2" in text:
            return "area"
        if any(term in text for term in _COUNT_TERMS):
            return "count"
        return "text"

    def _select_group(self, question: str, file_id: Optional[str]) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        if file_id is not None:
            entry = self.files.get(file_id)
            if entry is None:
                return None, {"status": "not_found", "reason": f"Unknown file_id: {file_id}", "evidence": []}
            self._active_group = entry.group_id
            return self._groups[entry.group_id], None
        if not self.files:
            return None, {"status": "not_found", "reason": "No files have been uploaded to this conversation.", "evidence": []}

        normalized = question.casefold()
        mentioned_groups = set()
        for entry in self.files.values():
            stem = entry.path.stem.casefold()
            if len(stem) >= 4 and (stem in normalized or entry.path.name.casefold() in normalized):
                mentioned_groups.add(entry.group_id)
        if len(mentioned_groups) == 1:
            group = next(iter(mentioned_groups))
            self._active_group = group
            return self._groups[group], None

        groups = list(dict.fromkeys(entry.group_id for entry in self.files.values()))
        if len(groups) == 1:
            self._active_group = groups[0]
            return self._groups[groups[0]], None
        if mentioned_groups:
            groups = [group for group in groups if group in mentioned_groups]
            if len(groups) == 1:
                self._active_group = groups[0]
                return self._groups[groups[0]], None
        follow_up = re.search(r"\b(?:it|its|this|that|there|same)\b", normalized)
        if follow_up and self._active_group in self._groups:
            return self._groups[self._active_group], None
        options = []
        for group in groups:
            members = [self.files[item] for item in self._groups[group]]
            representative = min(members, key=lambda item: _FORMAT_PRIORITY.get(item.format, 99))
            options.append({"file_id": representative.file_id, "name": representative.path.name,
                            "format": representative.format})
        return None, {
            "status": "clarification_required",
            "answer": None,
            "reason": "Several uploaded drawings could match; select the file to use.",
            "options": options,
            "evidence": [],
        }

    def _select_file(self, group_ids: List[str], intent: str) -> FileEntry:
        candidates = [self.files[file_id] for file_id in group_ids]
        supported = []
        for item in candidates:
            if item.format == "ifc" and intent == "area":
                supported.append((0, item))
            elif item.format == "dxf" and intent in {"area", "count"}:
                supported.append((1, item))
            elif item.format == "dwg" and intent in {"area", "count"}:
                supported.append((2, item))
            elif item.format == "pdf":
                supported.append((3, item))
            elif item.format == "image":
                supported.append((4, item))
            elif item.format == "ifc":
                supported.append((5, item))
        if supported:
            return min(supported, key=lambda pair: pair[0])[1]
        return min(candidates, key=lambda item: _FORMAT_PRIORITY.get(item.format, 99))

    def _parse(self, entry: FileEntry) -> Dict[str, Any]:
        with self._lock:
            if entry.parsed is not None:
                return entry.parsed
            try:
                if entry.format == "ifc":
                    from bim_parser import parse_ifc
                    parsed = parse_ifc(entry.path)
                elif entry.format == "dxf":
                    from cad_qa import build_floor_index
                    parsed = build_floor_index(entry.path)
                elif entry.format == "dwg":
                    from cad_qa import build_floor_index
                    parsed = build_floor_index(entry.path)
                elif entry.format == "pdf":
                    from pdf_parser import parse_pdf
                    parsed = parse_pdf(entry.path)
                elif entry.format == "image":
                    from image_parser import parse_image
                    parsed = parse_image(entry.path, run_ocr=True)
                    parsed["drawing_type"] = entry.drawing_type or "uncertain"
                else:
                    raise ValueError(f"Unsupported upload format: {entry.path.suffix or '(none)'}")
                entry.parsed = parsed
                entry.parse_status = "parsed"
                return parsed
            except Exception as exc:
                entry.parse_status = "error"
                entry.parse_error = f"{type(exc).__name__}: {exc}"
                raise

    def set_drawing_type(self, file_id: str, drawing_type: str) -> Dict[str, Any]:
        """Apply a user-confirmed image type and invalidate its parsed cache."""
        allowed = {"floor_plan", "elevation", "section", "detail", "site_plan", "map", "other", "uncertain"}
        if drawing_type not in allowed:
            raise ValueError(f"Unsupported drawing_type: {drawing_type}")
        with self._lock:
            entry = self.files.get(file_id)
            if entry is None:
                raise KeyError(f"Unknown file_id: {file_id}")
            if entry.format != "image":
                raise ValueError("Only uploaded raster images have a drawing type")
            entry.drawing_type = drawing_type
            entry.parsed = None
            entry.parse_status = "not_parsed"
            entry.parse_error = None
            return self._manifest_entry(file_id)

    def retry_parse(self, file_id: str) -> None:
        """Clear a cached parse failure so the caller can retry after setup changes."""
        with self._lock:
            entry = self.files.get(file_id)
            if entry is None:
                raise KeyError(f"Unknown file_id: {file_id}")
            entry.parsed = None
            entry.parse_status = "not_parsed"
            entry.parse_error = None

    def answer(self, question: str, *, file_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer a question from this chat's files, with grounded evidence."""
        groups, error = self._select_group(question, file_id)
        if error is not None:
            return error
        assert groups is not None
        intent = self._intent(question)
        entry = self._select_file(groups, intent)
        try:
            document = self._parse(entry)
        except Exception as exc:
            return {
                "status": "not_found", "answer": None,
                "reason": f"Could not parse {entry.path.name}: {type(exc).__name__}: {exc}",
                "evidence": [], "file_id": entry.file_id,
            }

        if entry.format == "ifc":
            if intent == "area":
                from bim_qa import answer_area_question
                result = answer_area_question(question, document)
            else:
                from ifc_qa import IfcIndex, answer_ifc_question, answer_ifc_text_question
                if entry.ifc_index is None:
                    entry.ifc_index = IfcIndex(str(entry.path))
                result = answer_ifc_question(question, entry.ifc_index)
                if result.get("status") == "unsupported":
                    result = answer_ifc_text_question(question, entry.ifc_index)
        elif entry.format in {"dxf", "dwg"}:
            if intent not in {"area", "count"}:
                result = {"status": "unsupported", "answer": None,
                          "reason": "The CAD answerer currently supports count and area questions.", "evidence": []}
            else:
                from cad_qa import answer_cad_question
                result = answer_cad_question(question, document)
        elif entry.format == "pdf":
            from pdf_parser import answer_pdf_question
            result = answer_pdf_question(question, document)
        elif entry.format == "image":
            from image_parser import answer_image_question
            result = answer_image_question(
                question, document, model_path=self.count_model, geometry_model_path=self.geometry_model,
                cubicasa_repo_path=self.cubicasa_repo,
                cubicasa_weights_path=self.cubicasa_weights,
            )
        else:
            result = {"status": "unsupported", "answer": None, "evidence": [],
                      "reason": "No question answerer is available for this file format."}

        result = dict(result)
        result["file_id"] = entry.file_id
        result["file"] = entry.path.name
        result["format"] = entry.format
        result["drawing_type"] = entry.drawing_type
        result["related_files"] = [self.files[item].path.name for item in groups if item != entry.file_id]
        return result

    def close(self) -> None:
        """Release this conversation's in-memory file metadata and parse cache."""
        with self._lock:
            self.files.clear()
            self._groups.clear()
            self._active_group = None
            self.classification_warnings.clear()
            self._vision_classifier = None
