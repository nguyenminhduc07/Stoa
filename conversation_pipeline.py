"""Conversation-scoped file ingestion, specialist evidence caching, and LLM-reasoned QA.

Create one ConversationPipeline per chat. Local parsers extract deterministic,
provenance-preserving EvidencePacks once per uploaded file, and the LLM reasons
over those inventories to select grounded evidence IDs and operations without
hardcoded question templates or keyword lists.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_reasoner import (
    EvidencePack,
    build_cad_evidence_pack,
    build_ifc_evidence_pack,
    build_image_evidence_pack,
    build_pdf_evidence_pack,
    execute_reasoning_decision,
    is_configured as llm_is_configured,
    request_reasoning_decision,
)


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_FORMAT_BY_EXTENSION = {
    ".ifc": "ifc",
    ".dxf": "dxf",
    ".dwg": "dwg",
    ".pdf": "pdf",
    **{extension: "image" for extension in _IMAGE_EXTENSIONS},
}
_FORMAT_PRIORITY = {"ifc": 0, "dxf": 1, "dwg": 1, "pdf": 2, "image": 3}


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
    rendered_pdf_pages: Dict[int, Dict[str, Any]] = field(default_factory=dict, repr=False)
    evidence_pack: Optional[EvidencePack] = field(default=None, repr=False)

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
    """Manage uploaded files, cached EvidencePacks, and LLM-reasoned grounded answers."""

    def __init__(
        self,
        *,
        count_model: Optional[str | Path] = None,
        geometry_model: Optional[str | Path] = None,
        cubicasa_repo: Optional[str | Path] = None,
        cubicasa_weights: Optional[str | Path] = None,
        object_detector_model: Optional[str | Path] = None,
        render_dir: Optional[str | Path] = None,
        reasoner_fn: Optional[Callable[..., Tuple[Dict[str, Any], str]]] = None,
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
        self.reasoner_fn = reasoner_fn
        self.files: Dict[str, FileEntry] = {}
        self._groups: Dict[str, List[str]] = {}
        self.history: List[Dict[str, Any]] = []
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
        """Identify file format from binary content signatures before falling back to suffix."""
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
        if (
            header.startswith(b"\x89PNG\r\n\x1a\n")
            or header.startswith(b"\xff\xd8")
            or header.startswith(b"BM")
            or header.startswith((b"II*\x00", b"MM\x00*"))
        ):
            return "image"
        return _FORMAT_BY_EXTENSION.get(suffix, "unsupported")

    def add_files(self, paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
        """Register uploaded paths and return their manifest entries."""
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
        """Use zero-shot CLIP labels when Transformers/model weights exist."""
        try:
            from PIL import Image
            from transformers import pipeline
            from file_classifier import DocumentRouter

            if self._vision_classifier is None:
                self._vision_classifier = pipeline(
                    task="zero-shot-image-classification",
                    model=DocumentRouter.IMAGE_MODEL,
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

    def _canonical_group(self, group_id: str) -> str:
        if "_plan-" in group_id:
            base = group_id.split("_plan-", 1)[0]
            if any(self.files[fid].format == "ifc" for fid in self._groups.get(base, [])):
                return base
        return group_id

    def _group_members_with_twins(self, group_id: str) -> List[str]:
        canon = self._canonical_group(group_id)
        members = list(self._groups.get(canon, []))
        for other_group, other_ids in self._groups.items():
            if other_group == group_id or (
                other_group.startswith(f"{canon}_plan-")
                and any(self.files[fid].format == "ifc" for fid in members)
            ):
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

    def set_drawing_type(self, file_id: str, drawing_type: str) -> Dict[str, Any]:
        """Apply a user-confirmed image type and invalidate its cached EvidencePack."""
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
            entry.evidence_pack = None
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
            entry.rendered_pdf_pages.clear()
            entry.evidence_pack = None
            entry.parse_status = "not_parsed"
            entry.parse_error = None

    def _render_pdf_visual_page(self, entry: FileEntry, page_num: int) -> None:
        """Render one PDF page at 150 DPI, run Image + Floor-Plan specialists, and update EvidencePack."""
        with self._lock:
            if page_num in entry.rendered_pdf_pages:
                return
            if entry.parsed is None:
                from pdf_parser import parse_pdf
                entry.parsed = parse_pdf(entry.path)
            pages = entry.parsed.get("pages") or []
            page_info = next((p for p in pages if int(p.get("page") or 1) == page_num), None)
            if page_info is None:
                return
            from pdf_parser import _render_pdf_page_document
            from image_parser import enrich_image_specialist_evidence

            img_doc = _render_pdf_page_document(
                entry.path, page_info, entry.parsed, render_dir=self.render_dir
            )
            enrich_image_specialist_evidence(
                img_doc,
                cubicasa_repo_path=self.cubicasa_repo,
                cubicasa_weights_path=self.cubicasa_weights,
                geometry_model_path=self.geometry_model,
            )
            p_w = float(page_info.get("width") or 1.0)
            p_h = float(page_info.get("height") or 1.0)
            i_w = max(1, int(img_doc.get("width") or 1))
            i_h = max(1, int(img_doc.get("height") or 1))
            entry.rendered_pdf_pages[page_num] = {
                "document": img_doc,
                "page_width": p_w,
                "page_height": p_h,
                "scale_x": p_w / i_w,
                "scale_y": p_h / i_h,
            }
            entry.evidence_pack = build_pdf_evidence_pack(
                entry.file_id,
                entry.path.name,
                entry.parsed,
                rendered_pages=entry.rendered_pdf_pages,
            )

    def _ensure_evidence_pack(self, entry: FileEntry, *, has_exact_twin: bool = False) -> EvidencePack:
        """Parse and cache the structured EvidencePack for one file."""
        with self._lock:
            if entry.evidence_pack is not None:
                return entry.evidence_pack
            try:
                if entry.format == "ifc":
                    from bim_parser import parse_ifc
                    from ifc_qa import IfcIndex
                    if entry.parsed is None:
                        entry.parsed = parse_ifc(entry.path)
                    if entry.ifc_index is None:
                        entry.ifc_index = IfcIndex(str(entry.path))
                    pack = build_ifc_evidence_pack(
                        entry.file_id, entry.path.name, entry.parsed, entry.ifc_index
                    )
                elif entry.format in {"dxf", "dwg"}:
                    from cad_qa import build_floor_index
                    from cad_parser import parse_cad
                    if entry.parsed is None:
                        entry.parsed = build_floor_index(entry.path)
                    if entry.generic_cad is None:
                        try:
                            entry.generic_cad = parse_cad(entry.path)
                        except Exception:
                            entry.generic_cad = None
                    pack = build_cad_evidence_pack(
                        entry.file_id, entry.path.name, entry.format, entry.parsed, entry.generic_cad
                    )
                elif entry.format == "pdf":
                    from pdf_parser import parse_pdf
                    if entry.parsed is None:
                        entry.parsed = parse_pdf(entry.path)
                    pages = entry.parsed.get("pages") or []
                    # Pre-render single-sheet standalone drawing PDFs with no schedule tables
                    # so spatial floor-plan evidence is available on the very first turn.
                    if not has_exact_twin and len(pages) == 1 and not (pages[0].get("tables") or []):
                        p0 = pages[0]
                        vec_cnt = (
                            int(p0.get("vector_line_count") or 0)
                            + int(p0.get("vector_rect_count") or 0)
                            + int(p0.get("image_count") or 0)
                        )
                        if vec_cnt > 0:
                            self._render_pdf_visual_page(entry, 1)
                    pack = entry.evidence_pack or build_pdf_evidence_pack(
                        entry.file_id,
                        entry.path.name,
                        entry.parsed,
                        rendered_pages=entry.rendered_pdf_pages,
                    )
                elif entry.format == "image":
                    from image_parser import enrich_image_specialist_evidence, parse_image
                    if entry.parsed is None:
                        entry.parsed = parse_image(entry.path, run_ocr=True)
                    entry.parsed["drawing_type"] = entry.drawing_type or "uncertain"
                    enrich_image_specialist_evidence(
                        entry.parsed,
                        cubicasa_repo_path=self.cubicasa_repo,
                        cubicasa_weights_path=self.cubicasa_weights,
                        geometry_model_path=self.geometry_model,
                    )
                    pack = build_image_evidence_pack(entry.file_id, entry.path.name, entry.parsed)
                else:
                    raise ValueError(f"Unsupported upload format: {entry.path.suffix or '(none)'}")

                entry.evidence_pack = pack
                entry.parse_status = "parsed"
                entry.parse_error = None
                return pack
            except Exception as exc:
                entry.parse_status = "error"
                entry.parse_error = f"{type(exc).__name__}: {exc}"
                raise

    def _candidate_entries(self, forced_file_id: Optional[str]) -> List[FileEntry]:
        """Select candidate files per drawing group, prioritizing exact formats for twins."""
        if forced_file_id is not None:
            forced = self.files.get(forced_file_id)
            if forced is None:
                return []
            twins = [
                self.files[fid]
                for fid in self._group_members_with_twins(forced.group_id)
                if fid in self.files and fid != forced_file_id
            ]
            twins.sort(key=lambda item: _FORMAT_PRIORITY.get(item.format, 99))
            return [forced] + twins[:1]

        canonical_groups = list(
            dict.fromkeys(self._canonical_group(e.group_id) for e in self.files.values())
        )
        selected: List[FileEntry] = []
        for group in canonical_groups:
            members = [self.files[fid] for fid in self._group_members_with_twins(group) if fid in self.files]
            members.sort(key=lambda item: _FORMAT_PRIORITY.get(item.format, 99))
            if not members:
                continue
            primary = members[0]
            selected.append(primary)
            # If the primary twin is IFC/DXF and there is also a PDF schedule twin, include the PDF too.
            if primary.format in {"ifc", "dxf", "dwg"}:
                pdf_twin = next((m for m in members[1:] if m.format == "pdf" and "_plan-" not in m.group_id), None)
                if pdf_twin is not None:
                    selected.append(pdf_twin)
        return selected[:8]

    @staticmethod
    def _validate_answer(result: Dict[str, Any]) -> Dict[str, Any]:
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

        unit = validated.get("unit")
        if isinstance(primary, (int, float)) and not isinstance(primary, bool):
            if not math.isfinite(float(primary)):
                return {
                    "status": "not_found",
                    "answer": None,
                    "reason": "Answer validation rejected a non-finite numeric quantity.",
                    "evidence": [],
                    "validated": False,
                }
            if unit in {"count", "m2", "m", "m3"} and float(primary) < 0:
                return {
                    "status": "not_found",
                    "answer": None,
                    "reason": "Answer validation rejected a negative physical quantity.",
                    "evidence": [],
                    "validated": False,
                }
            if unit == "count":
                validated["value"] = int(round(float(primary)))
                validated["answer"] = validated["value"]

        evidence = validated.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            validated["review_recommended"] = True
            validated["validation_warning"] = "Answer lacked structured location citations."
            validated["validated"] = False
        else:
            validated["validated"] = True
        return validated

    def answer(self, question: str, *, file_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer any natural-language question via grounded LLM reasoning over cached EvidencePacks."""
        if not self.files:
            return {
                "status": "not_found",
                "answer": None,
                "reason": "No files have been uploaded to this conversation.",
                "evidence": [],
            }
        if file_id is not None and file_id not in self.files:
            return {
                "status": "not_found",
                "answer": None,
                "reason": f"Unknown file_id: {file_id}",
                "evidence": [],
            }

        reasoner = self.reasoner_fn or request_reasoning_decision
        if self.reasoner_fn is None and not llm_is_configured():
            return {
                "status": "unsupported",
                "answer": None,
                "reason": (
                    "Natural-language reasoning requires an LLM API key. "
                    "Set GEMINI_API_KEY in .env (or export GEMINI_API_KEY in your shell) and ask again."
                ),
                "evidence": [],
                "query_interpretation": {
                    "planner": "llm_grounded_reasoner",
                    "status": "not_configured",
                },
            }

        candidates = self._candidate_entries(file_id)
        packs: List[EvidencePack] = []
        last_parse_error: Optional[Tuple[FileEntry, Exception]] = None
        for entry in candidates:
            siblings = self._group_members_with_twins(entry.group_id)
            has_exact_twin = any(
                self.files[fid].format in {"ifc", "dxf", "dwg"} and fid != entry.file_id
                for fid in siblings
                if fid in self.files
            )
            try:
                packs.append(self._ensure_evidence_pack(entry, has_exact_twin=has_exact_twin))
            except Exception as exc:
                last_parse_error = (entry, exc)

        if not packs:
            if last_parse_error is not None:
                failed_entry, exc = last_parse_error
                return {
                    "status": "not_found",
                    "answer": None,
                    "reason": f"Could not parse {failed_entry.path.name}: {type(exc).__name__}: {exc}",
                    "evidence": [],
                    "file_id": failed_entry.file_id,
                }
            return {
                "status": "not_found",
                "answer": None,
                "reason": "No readable drawing files are available in this conversation.",
                "evidence": [],
            }

        twin_map = {
            entry.file_id: [
                self.files[fid].path.name
                for fid in self._group_members_with_twins(entry.group_id)
                if fid in self.files and fid != entry.file_id
            ]
            for entry in candidates
            if entry.file_id in {p.file_id for p in packs}
        }

        try:
            decision, provider = reasoner(
                question,
                packs,
                history=self.history,
                forced_file_id=file_id,
                twin_relationships=twin_map,
            )

            # On-demand visual page render loop for PDFs when visual analysis is requested by the LLM
            if str(decision.get("action")) == "render_pdf_page":
                target_fid = str(decision.get("file_id") or packs[0].file_id)
                target_entry = self.files.get(target_fid)
                target_page = int(decision.get("pdf_page_to_render") or 1)
                if target_entry is not None and target_entry.format == "pdf":
                    self._render_pdf_visual_page(target_entry, target_page)
                    if target_entry.evidence_pack is not None:
                        packs = [
                            target_entry.evidence_pack if p.file_id == target_fid else p
                            for p in packs
                        ]
                        decision, provider = reasoner(
                            question,
                            packs,
                            history=self.history,
                            forced_file_id=file_id,
                            twin_relationships=twin_map,
                        )
        except Exception as exc:
            return {
                "status": "not_found",
                "answer": None,
                "reason": f"LLM reasoning request failed ({type(exc).__name__}: {exc}). Check GEMINI_API_KEY and network connectivity.",
                "evidence": [],
                "query_interpretation": {
                    "planner": "llm_grounded_reasoner",
                    "status": "error",
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                },
            }

        packs_by_id = {p.file_id: p for p in packs}
        clarification_options = (
            [
                {"file_id": p.file_id, "name": p.file_name, "format": p.format}
                for p in packs
            ]
            if len(packs) > 1
            else None
        )
        chosen_pack, raw_result = execute_reasoning_decision(
            decision,
            packs_by_id,
            provider=provider,
            fallback_options=clarification_options,
        )
        validated = self._validate_answer(raw_result)

        chosen_entry = self.files.get(chosen_pack.file_id) if chosen_pack else self.files.get(packs[0].file_id)
        if chosen_entry is not None:
            validated["file_id"] = chosen_entry.file_id
            validated["file"] = chosen_entry.path.name
            validated["format"] = chosen_entry.format
            validated["drawing_type"] = chosen_entry.drawing_type
            siblings = self._group_members_with_twins(chosen_entry.group_id)
            validated["related_files"] = [
                self.files[fid].path.name
                for fid in siblings
                if fid in self.files and fid != chosen_entry.file_id
            ]

        validated["query_interpretation"] = {
            "planner": f"llm_grounded_reasoner ({provider})",
            "status": "reasoned",
            "operation": decision.get("operation"),
            "selected_count": len(decision.get("selected_ids") or []),
        }

        if validated.get("status") == "answered":
            self.history.append({
                "question": question,
                "file_id": validated.get("file_id"),
                "file": validated.get("file"),
                "answer": validated.get("value", validated.get("answer")),
                "unit": validated.get("unit"),
                "explanation": validated.get("reason"),
            })
            if len(self.history) > 12:
                self.history = self.history[-12:]

        return validated

    def close(self) -> None:
        """Release this conversation's in-memory file metadata and parse cache."""
        with self._lock:
            self.files.clear()
            self._groups.clear()
            self.history.clear()
            self.classification_warnings.clear()
            self._vision_classifier = None
