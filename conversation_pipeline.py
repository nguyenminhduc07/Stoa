"""Conversation-scoped file ingestion, parser caching, and grounded question routing.

Create one ConversationPipeline per chat. It holds only in-memory metadata and
parsed results; callers should discard it when that conversation ends.
"""

from __future__ import annotations

import hashlib
import math
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
_LENGTH_TERMS = ("length", "laengde", "wie lang", "pikkus", "longueur", "pituus")
_VOLUME_TERMS = ("volume", "volumen", "maht", "tilavuus")
_COUNT_TERMS = ("how many", "number of", "count", "combien", "nombre de", "wie viele", "anzahl",
                "quante", "quanti", "cuantos", "cuantas", "kuinka monta", "montako", "hvor mange", "mitu", "how much")
_INTENT_BY_PLAN = {
    "count": "count", "area": "area", "length": "length", "volume": "volume",
    "lookup": "text", "describe": "text", "compare": "text", "other": "text",
}


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
    generic_cad: Optional[Dict[str, Any]] = field(default=None, repr=False)

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
        object_detector_model: Optional[str | Path] = None,
        render_dir: Optional[str | Path] = None,
    ) -> None:
        if cubicasa_repo is None and cubicasa_weights is None:
            try:
                from image_parser import default_cubicasa_paths
                default_repo, default_weights = default_cubicasa_paths()
            except Exception:
                default_repo, default_weights = None, None
        else:
            default_repo, default_weights = None, None
        self.count_model = Path(count_model) if count_model else None
        self.geometry_model = Path(geometry_model) if geometry_model else None
        self.cubicasa_repo = Path(cubicasa_repo) if cubicasa_repo else default_repo
        self.cubicasa_weights = Path(cubicasa_weights) if cubicasa_weights else default_weights
        self.object_detector_model = object_detector_model
        self.render_dir = Path(render_dir) if render_dir else None
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
        if header.startswith(b"AutoCAD Binary DXF") or re.search(rb"(?:^|\r?\n)\s*0\r?\nSECTION\r?\n", header):
            return "dxf"
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

    def _group_members_with_twins(self, group_id: str) -> List[str]:
        members = list(self._groups.get(group_id, []))
        if "_plan-" in group_id:
            base_group = group_id.split("_plan-", 1)[0]
            ifc_twins = [
                fid for fid in self._groups.get(base_group, [])
                if self.files[fid].format == "ifc" and fid not in members
            ]
            members = ifc_twins + members
        for other_group, other_ids in self._groups.items():
            if other_group.startswith(f"{group_id}_plan-"):
                if any(self.files[fid].format == "ifc" for fid in members):
                    for fid in other_ids:
                        if fid not in members:
                            members.append(fid)
        return members

    def _manifest_entry(self, file_id: str) -> Dict[str, Any]:
        entry = self.files[file_id]
        related = [item for item in self._group_members_with_twins(entry.group_id) if item != file_id]
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
        if any(term in text for term in _VOLUME_TERMS) or "m3" in text:
            return "volume"
        if any(term in text for term in _LENGTH_TERMS):
            return "length"
        if any(term in text for term in _COUNT_TERMS):
            return "count"
        return "text"

    def _canonical_group(self, group_id: str) -> str:
        if "_plan-" in group_id:
            base = group_id.split("_plan-", 1)[0]
            if any(self.files[fid].format == "ifc" for fid in self._groups.get(base, [])):
                return base
        return group_id

    def _select_group(self, question: str, file_id: Optional[str]) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        if file_id is not None:
            entry = self.files.get(file_id)
            if entry is None:
                return None, {"status": "not_found", "reason": f"Unknown file_id: {file_id}", "evidence": []}
            self._active_group = self._canonical_group(entry.group_id)
            members = self._group_members_with_twins(entry.group_id)
            return [file_id] + [fid for fid in members if fid != file_id], None
        if not self.files:
            return None, {"status": "not_found", "reason": "No files have been uploaded to this conversation.", "evidence": []}

        normalized = question.casefold()
        mentioned_groups = set()
        for entry in self.files.values():
            stem = entry.path.stem.casefold()
            if len(stem) >= 4 and (stem in normalized or entry.path.name.casefold() in normalized):
                mentioned_groups.add(self._canonical_group(entry.group_id))
        if len(mentioned_groups) == 1:
            group = next(iter(mentioned_groups))
            self._active_group = group
            return self._group_members_with_twins(group), None

        groups = list(dict.fromkeys(self._canonical_group(entry.group_id) for entry in self.files.values()))
        if len(groups) == 1:
            self._active_group = groups[0]
            return self._group_members_with_twins(groups[0]), None
        if mentioned_groups:
            groups = [group for group in groups if group in mentioned_groups]
            if len(groups) == 1:
                self._active_group = groups[0]
                return self._group_members_with_twins(groups[0]), None
        follow_up = re.search(r"\b(?:it|its|this|that|there|same)\b", normalized)
        if follow_up and self._active_group in self._groups:
            return self._group_members_with_twins(self._active_group), None
        options = []
        for group in groups:
            members = [self.files[item] for item in self._group_members_with_twins(group)]
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

    def _rank_files(self, group_ids: List[str], intent: str, forced_file_id: Optional[str] = None) -> List[FileEntry]:
        candidates = [self.files[fid] for fid in group_ids if fid in self.files]
        if forced_file_id and any(item.file_id == forced_file_id for item in candidates):
            forced = next(item for item in candidates if item.file_id == forced_file_id)
            rest = [item for item in candidates if item.file_id != forced_file_id]
            rest.sort(key=lambda item: _FORMAT_PRIORITY.get(item.format, 99))
            return [forced] + rest

        def score(item: FileEntry) -> Tuple[int, int]:
            fmt = item.format
            if fmt == "ifc" and intent in {"area", "length", "volume", "count", "text"}:
                rank = 0
            elif fmt == "dxf" and intent in {"area", "count"}:
                rank = 1
            elif fmt == "dwg" and intent in {"area", "count"}:
                rank = 2
            elif fmt == "pdf":
                rank = 3
            elif fmt in {"dxf", "dwg"}:
                rank = 4
            elif fmt == "image":
                rank = 5
            elif fmt == "ifc":
                rank = 6
            else:
                rank = 99
            return (rank, _FORMAT_PRIORITY.get(fmt, 99))

        return sorted(candidates, key=score)

    def _select_file(self, group_ids: List[str], intent: str) -> FileEntry:
        return self._rank_files(group_ids, intent)[0]

    def _parse(self, entry: FileEntry) -> Dict[str, Any]:
        with self._lock:
            if entry.parsed is not None:
                return entry.parsed
            try:
                if entry.format == "ifc":
                    from bim_parser import parse_ifc
                    parsed = parse_ifc(entry.path)
                elif entry.format in {"dxf", "dwg"}:
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

    def _parse_generic_cad(self, entry: FileEntry) -> Dict[str, Any]:
        with self._lock:
            if entry.generic_cad is None:
                from cad_parser import parse_cad
                entry.generic_cad = parse_cad(entry.path)
            return entry.generic_cad

    @staticmethod
    def _answer_generic_cad_question(question: str, cad_doc: Dict[str, Any]) -> Dict[str, Any]:
        q_norm = ConversationPipeline._normalise(question)
        source_path = cad_doc.get("source", {}).get("path")
        entities = cad_doc.get("entities", [])

        if any(term in q_norm for term in ("layer", "layers", "what layers")):
            layers = [layer for layer in cad_doc.get("layers", []) if layer.get("entity_count", 0) > 0]
            if layers:
                summary = ", ".join(f"{item['name']} ({item['entity_count']})" for item in layers[:25])
                return {
                    "status": "answered",
                    "answer": f"{len(layers)} populated CAD layers: {summary}",
                    "method": "cad_layer_inventory",
                    "evidence": [{"source": source_path, "layers": [item["name"] for item in layers[:25]]}],
                }

        quoted = re.search(r"[\"'](.+?)[\"']|[\u2018\u201c](.+?)[\u2019\u201d]", question)
        if quoted:
            target = ConversationPipeline._normalise(quoted.group(1) or quoted.group(2)).strip()
            matches = [
                ent for ent in entities
                if ent.get("text") and target and target in ConversationPipeline._normalise(ent["text"])
            ]
            if matches:
                first = matches[0]
                return {
                    "status": "answered",
                    "answer": str(first["text"]),
                    "method": "cad_text_entity_lookup",
                    "evidence": [
                        {
                            "source": source_path,
                            "handle": ent.get("handle"),
                            "layer": ent.get("layer"),
                            "bbox": ent.get("bbox"),
                            "text": ent.get("text"),
                        }
                        for ent in matches[:10]
                    ],
                }

        if any(term in q_norm for term in ("dimension", "dimensions", "measurement", "measurements")):
            dims = [
                ent for ent in entities
                if ent.get("type") == "DIMENSION" and (
                    ent.get("geometry", {}).get("measurement") is not None or ent.get("text")
                )
            ]
            if dims:
                unit_name = cad_doc.get("units", {}).get("name") or "drawing units"
                items_txt = [
                    str(ent.get("text") or f"{round(float(ent['geometry']['measurement']), 3)} {unit_name}")
                    for ent in dims[:10]
                ]
                return {
                    "status": "answered",
                    "answer": ", ".join(items_txt),
                    "method": "cad_dimension_entities",
                    "evidence": [
                        {
                            "source": source_path,
                            "handle": ent.get("handle"),
                            "layer": ent.get("layer"),
                            "bbox": ent.get("bbox"),
                            "measurement": ent.get("geometry", {}).get("measurement"),
                        }
                        for ent in dims[:10]
                    ],
                }

        return {
            "status": "unsupported",
            "answer": None,
            "reason": "The CAD answerer could not match this question to semantic floor-plan layers or CAD text/dimension entities.",
            "evidence": [],
        }

    @staticmethod
    def _filters_supported_for_format(fmt: str, intent: str, plan: Any) -> bool:
        if plan is None or not getattr(plan, "filters", ()):
            return True
        universal_cues = (
            "indoor", "interior room", "numbered", "outdoor", "balcon", "loggia",
            "terrac", "shaft", "void", "separate", "storey", "floor", "level",
            "korrus", "sal", "geschoss", "etage", "plan", "drawing", "whole", "entire",
        )
        ifc_cues = (
            "exterior", "external", "interior", "internal", "partition",
            "load-bearing", "load bearing", "loadbearing", "concrete", "foundation",
            "structural", "fire", "glazing", "thermal", "u-value", "u value",
            "office", "buero", "tile", "tiling", "vapor", "vapour", "skirting",
            "type", "material", "room", "space", "door", "window", "pipe", "duct",
        )
        cad_cues = ("apartment", "wohnung", "appartement", "unit", "whg", "room", "label", "named", "number")
        pdf_cues = (
            "schedule", "type", "mark", "material", "frame", "finish", "fire",
            "hardware", "north", "south", "east", "west", "room", "door", "window", "note", "page",
        )
        target = str(getattr(plan, "target", "") or "")
        for raw_filter in plan.filters:
            norm_f = ConversationPipeline._normalise(str(raw_filter))
            if any(cue in norm_f for cue in universal_cues):
                continue
            if fmt == "ifc" and any(cue in norm_f for cue in ifc_cues):
                continue
            if fmt in {"dxf", "dwg"} and (intent == "area" or target in {"rooms", "spaces", "other"}) and any(
                cue in norm_f for cue in cad_cues
            ):
                continue
            if fmt == "pdf" and any(cue in norm_f for cue in pdf_cues):
                continue
            return False
        return True

    @staticmethod
    def _validate_answer(result: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """Validate numeric sanity and provenance citations before returning an answer."""
        validated = dict(result)
        status = validated.get("status")
        if status != "answered":
            validated["validated"] = True
            return validated

        primary = validated.get("value", validated.get("answer"))
        if primary is None or (isinstance(primary, str) and not primary.strip()):
            return {
                "status": "not_found",
                "answer": None,
                "reason": "Answer validation rejected an empty answer payload.",
                "evidence": [],
                "validated": False,
            }

        if isinstance(primary, (int, float)) and not isinstance(primary, bool):
            if not math.isfinite(float(primary)):
                return {
                    "status": "not_found",
                    "answer": None,
                    "reason": "Answer validation rejected a non-finite numeric quantity.",
                    "evidence": [],
                    "validated": False,
                }
            if intent in {"count", "area", "length", "volume"} and float(primary) < 0:
                return {
                    "status": "not_found",
                    "answer": None,
                    "reason": "Answer validation rejected a negative physical quantity.",
                    "evidence": [],
                    "validated": False,
                }
            if intent == "count":
                validated["value"] = int(round(float(primary)))

        evidence = validated.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            validated["review_recommended"] = True
            validated["validation_warning"] = "Answer lacked structured location citations."
            validated["validated"] = False
        else:
            validated["validated"] = True
        return validated

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
            entry.generic_cad = None
            entry.parse_status = "not_parsed"
            entry.parse_error = None

    def _answer_with_entry(
        self,
        entry: FileEntry,
        parser_question: str,
        intent: str,
        plan: Any,
    ) -> Dict[str, Any]:
        if not self._filters_supported_for_format(entry.format, intent, plan):
            return {
                "status": "unsupported",
                "answer": None,
                "reason": (
                    "The question includes a condition the current drawing query executors do not "
                    "apply yet, so I am not returning an unfiltered result."
                ),
                "evidence": [],
            }

        document = self._parse(entry)
        if entry.format == "ifc":
            from ifc_qa import IfcIndex, answer_ifc_question, answer_ifc_text_question
            if entry.ifc_index is None:
                entry.ifc_index = IfcIndex(str(entry.path))
            has_ifc_filter = bool(plan and getattr(plan, "filters", ())) and any(
                cue in self._normalise(" ".join(plan.filters))
                for cue in ("exterior", "external", "interior", "internal", "partition", "load", "concrete", "type", "material")
            )
            if intent == "area" and not has_ifc_filter:
                result = answer_ifc_question(parser_question, entry.ifc_index)
                if result.get("status") == "unsupported":
                    from bim_qa import answer_area_question
                    result = answer_area_question(parser_question, document)
                if result.get("status") in {"unsupported", "not_found"}:
                    text_res = answer_ifc_text_question(parser_question, entry.ifc_index)
                    if text_res.get("status") == "answered":
                        result = text_res
                return result

            text_first = intent == "text" or has_ifc_filter or (
                intent == "count" and any(term in parser_question.casefold() for term in ("type", "types", "breakdown", "distribution"))
            )
            if text_first:
                result = answer_ifc_text_question(parser_question, entry.ifc_index)
                if result.get("status") == "unsupported" and not has_ifc_filter:
                    result = answer_ifc_question(parser_question, entry.ifc_index)
            else:
                result = answer_ifc_question(parser_question, entry.ifc_index)
                if result.get("status") == "unsupported":
                    result = answer_ifc_text_question(parser_question, entry.ifc_index)
            return result

        if entry.format in {"dxf", "dwg"}:
            has_semantic_layers = any(
                bool(document.get(key)) for key in ("rooms", "units", "doors", "windows", "fixtures")
            )
            if intent in {"area", "count"} and has_semantic_layers:
                from cad_qa import answer_cad_question
                cad_res = answer_cad_question(parser_question, document)
                if cad_res.get("status") == "answered":
                    return cad_res
                if intent in {"area", "count"}:
                    return cad_res
            cad_doc = self._parse_generic_cad(entry)
            return self._answer_generic_cad_question(parser_question, cad_doc)

        if entry.format == "pdf":
            from pdf_parser import answer_pdf_question
            return answer_pdf_question(
                parser_question,
                document,
                enable_visual_fallback=True,
                render_dir=self.render_dir,
                model_path=self.count_model,
                geometry_model_path=self.geometry_model,
                cubicasa_repo_path=self.cubicasa_repo,
                cubicasa_weights_path=self.cubicasa_weights,
                object_detector_model=self.object_detector_model,
                query_plan=plan.public() if plan else None,
            )

        if entry.format == "image":
            from image_parser import answer_image_question
            return answer_image_question(
                parser_question,
                document,
                model_path=self.count_model,
                geometry_model_path=self.geometry_model,
                cubicasa_repo_path=self.cubicasa_repo,
                cubicasa_weights_path=self.cubicasa_weights,
                object_detector_model=self.object_detector_model,
                query_plan=plan.public() if plan else None,
            )

        return {
            "status": "unsupported",
            "answer": None,
            "evidence": [],
            "reason": "No question answerer is available for this file format.",
        }

    def answer(self, question: str, *, file_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer a question from this chat's files, with grounded evidence."""
        groups, error = self._select_group(question, file_id)
        if error is not None:
            return error
        assert groups is not None
        plan = None
        planner_error = None
        try:
            from llm_query_planner import plan_question
            group_formats = [self.files[item].format for item in groups]
            plan = plan_question(question, group_formats)
        except Exception as exc:
            # The LLM improves interpretation but must never prevent the
            # existing deterministic route from answering a question.
            planner_error = type(exc).__name__
        intent = _INTENT_BY_PLAN.get(plan.intent, "text") if plan else self._intent(question)
        parser_question = plan.canonical_question if plan else question
        interpretation = {
            "planner": "openai_function_call" if plan else "keyword_fallback",
            "status": "planned" if plan else ("fallback" if planner_error else "not_configured"),
            "intent": plan.intent if plan else intent,
            "target": plan.target if plan else None,
            "scope": plan.scope if plan else None,
            "fallback_reason": planner_error,
        }
        if plan and plan.clarification_question:
            return {
                "status": "clarification_required", "answer": plan.clarification_question,
                "reason": "The question needs one more detail before it can be checked against the drawing.",
                "evidence": [], "query_interpretation": interpretation,
            }

        ranked_entries = self._rank_files(groups, intent, forced_file_id=file_id)
        chosen_entry = ranked_entries[0]
        chosen_result: Optional[Dict[str, Any]] = None
        last_parse_error: Optional[Tuple[FileEntry, Exception]] = None

        for candidate in ranked_entries:
            try:
                candidate_result = self._answer_with_entry(candidate, parser_question, intent, plan)
            except Exception as exc:
                last_parse_error = (candidate, exc)
                if file_id is not None and candidate.file_id == file_id:
                    break
                continue

            validated = self._validate_answer(candidate_result, intent)
            if chosen_result is None:
                chosen_entry = candidate
                chosen_result = validated
            if validated.get("status") in {"answered", "clarification_required"}:
                chosen_entry = candidate
                chosen_result = validated
                break
            # If the user explicitly forced a single file_id, do not silently switch files.
            if file_id is not None and candidate.file_id == file_id:
                break

        if chosen_result is None and last_parse_error is not None:
            failed_entry, exc = last_parse_error
            return {
                "status": "not_found", "answer": None,
                "reason": f"Could not parse {failed_entry.path.name}: {type(exc).__name__}: {exc}",
                "evidence": [], "file_id": failed_entry.file_id,
                "query_interpretation": interpretation,
            }

        assert chosen_result is not None
        result = dict(chosen_result)
        result["file_id"] = chosen_entry.file_id
        result["file"] = chosen_entry.path.name
        result["format"] = chosen_entry.format
        result["drawing_type"] = chosen_entry.drawing_type
        result["related_files"] = [self.files[item].path.name for item in groups if item != chosen_entry.file_id]
        result["query_interpretation"] = interpretation
        return result

    def close(self) -> None:
        """Release this conversation's in-memory file metadata and parse cache."""
        with self._lock:
            self.files.clear()
            self._groups.clear()
            self._active_group = None
            self.classification_warnings.clear()
            self._vision_classifier = None
