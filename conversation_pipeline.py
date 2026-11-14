"""Conversation-scoped file ingestion, specialist evidence caching, and 2-tier QA.

Create one ConversationPipeline per chat. Uploaded files are identified from
binary content signatures, grouped with their exact format twins (IFC > DXF/DWG >
PDF > raster image), and answered via:
  - Tier 1: Fast, deterministic local specialist parsers (IFC, CAD, PDF schedules
    + on-demand rendered visual fallback, and Image OCR + CubiCasa5K segmentation).
  - Tier 2: Grounded LLM compositional reasoning over structured EvidencePacks
    when questions combine/exclude subsets, reference prior turns, disambiguate
    files, or go beyond deterministic query rules.
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

_HARD_ABSTENTION_MARKERS = (
    "no shower class",
    "missing authored",
    "incomplete",
    "cannot be answered reliably",
)


def _is_ifc_text_query(question: str) -> bool:
    q = question.casefold()
    return any(
        term in q
        for term in (
            "breakdown", "of each", "each type", "which ", "list ", "compared with",
            "summarize", "summary", "describe", "fire rating", "fire resistance",
            "u-value", "u value", "thermal", "material", "thickness",
            "floor-to-floor", "floor to floor", "heights of each", "elevations of",
            "level elevations", "widest", "narrowest", "minimum", "maximum",
            "average", "program", "schema", "tiling", "ceiling", "insulation",
            "load-bearing", "load bearing", "footprint area", "each floor",
            "each room", "vertical circulation",
        )
    )


def _is_ifc_area_query(question: str) -> bool:
    q = question.casefold()
    if any(skip in q for skip in ("window", "glazed", "tile", "covering", "side area", "seitenfläche", "seitenflache")):
        return False
    return any(
        term in q
        for term in (
            "area", "fläche", "flache", "surface", "superficie",
            "pinta-ala", "pindala", "areal", "netopõrandapind", "nettoareal", "m²", "m2",
        )
    )


def _image_question_needs_ocr(question: str) -> bool:
    q = question.casefold()
    if any(
        term in q
        for term in (
            "read", "label", "text", "title", "scale", "drawing number", "sheet",
            "project", "date", "dimension", "width", "length", "height", "wc",
            "sauna", "löyly", "loyly", "parveke", "balcon", "storey", "kerros",
            "floor", "etage", "étage", "geschoss", "korrus",
        )
    ):
        return True
    try:
        from train_floorplan_model import question_category
        cat = question_category(question)
        if cat in {"wc_rooms", "saunas", "balconies", "storeys"}:
            return True
    except Exception:
        pass
    return False


def _needs_compositional_llm(question: str, *, has_history: bool = False) -> bool:
    """Detect compositional exclusions, cross-category sums, or conversational follow-ups."""
    q = question.casefold().strip()
    # Standard CAD/BIM prompts specify "(excluding balconies, loggias, terraces...)" which
    # deterministic cad_qa/bim_qa already implement natively; only bypass Tier 1 when
    # excluding indoor room/element types.
    excl_match = re.search(r"\b(?:excluding|except|without|other than|minus)\s+([a-zäöüõšž\s,/-]+)", q)
    if excl_match:
        tail = excl_match.group(1)
        if not any(out in tail for out in ("balcon", "loggia", "terrace", "outdoor", "exterior", "espaces ext")):
            return True

    # Cross-type compositional sums on floor plans (e.g., "doors and windows", "bedrooms plus bathrooms")
    if re.search(r"\bhow\s+many\b.*\b(?:doors?\s+(?:and|plus|\+)\s+windows?|windows?\s+(?:and|plus|\+)\s+doors?)\b", q):
        return True
    if re.search(r"\bhow\s+many\b.*\b(?:bedrooms?\s+(?:and|plus|\+)\s+(?:bathrooms?|kitchens?)|toilets?\s+(?:and|plus|\+)\s+sinks?)\b", q):
        return True

    if has_history and re.match(r"^(?:what about|how about|and (?:in|on|without|excluding)|of those|among those|between them)\b", q):
        return True
    return False


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
    """Manage uploaded files, cached specialist evidence, and hybrid deterministic + LLM QA."""

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
        default_repo, default_weights = None, None
        if cubicasa_weights is None:
            try:
                from image_parser import default_cubicasa_paths
                default_repo, default_weights = default_cubicasa_paths()
            except Exception:
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
        self._active_group: Optional[str] = None
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
        with self._lock:
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
        """Classify an uploaded raster image via dataset provenance, CLIP, and structural geometry."""
        if any(part.casefold() in {"cubicasa5k", "swiss_dwellings"} for part in path.parts):
            return "floor_plan"

        clip_type = "uncertain"
        best_score = 0.0
        try:
            from PIL import Image, ImageOps
            from file_classifier import DocumentRouter

            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")

            if self._vision_classifier is None:
                try:
                    from transformers import CLIPModel, CLIPProcessor
                    try:
                        clip_model = CLIPModel.from_pretrained(DocumentRouter.IMAGE_MODEL, use_safetensors=True)
                    except TypeError:
                        clip_model = CLIPModel.from_pretrained(DocumentRouter.IMAGE_MODEL)
                    try:
                        clip_proc = CLIPProcessor.from_pretrained(DocumentRouter.IMAGE_MODEL, use_fast=False)
                    except TypeError:
                        clip_proc = CLIPProcessor.from_pretrained(DocumentRouter.IMAGE_MODEL)
                    clip_model.eval()
                    self._vision_classifier = ("clip", (clip_model, clip_proc))
                except Exception:
                    from transformers import pipeline
                    self._vision_classifier = (
                        "pipeline",
                        pipeline(
                            task="zero-shot-image-classification",
                            model=DocumentRouter.IMAGE_MODEL,
                            device=-1,
                        ),
                    )

            scores_by_type = DocumentRouter.score_clip_image(self._vision_classifier, image)
            if scores_by_type:
                ranked = sorted(scores_by_type.items(), key=lambda kv: kv[1], reverse=True)
                best_type, best_score = ranked[0]
                second_score = ranked[1][1] if len(ranked) > 1 else 0.0
                margin = best_score - second_score
                if best_score >= 0.24 and margin >= 0.02:
                    clip_type = best_type
                else:
                    self.classification_warnings[file_id] = (
                        f"Low CLIP margin: {best_type}={best_score:.3f}, runner-up={second_score:.3f}."
                    )
        except Exception as exc:
            self.classification_warnings[file_id] = f"{type(exc).__name__}: {exc}"

        if clip_type == "floor_plan":
            self.classification_warnings.pop(file_id, None)
            return "floor_plan"
        if clip_type in {"elevation", "section"} and best_score >= 0.32:
            self.classification_warnings.pop(file_id, None)
            return clip_type

        # Structural & geometric floor-plan verification (CubiCasa5K segmentation or wall-topology watershed)
        try:
            from image_parser import is_likely_floorplan_by_structure
            if (clip_type == "uncertain" or best_score < 0.65) and is_likely_floorplan_by_structure(
                path,
                cubicasa_repo_path=self.cubicasa_repo,
                cubicasa_weights_path=self.cubicasa_weights,
            ):
                self.classification_warnings.pop(file_id, None)
                return "floor_plan"
        except Exception:
            pass

        if clip_type != "uncertain":
            self.classification_warnings.pop(file_id, None)
        return clip_type

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
        with self._lock:
            return [self._manifest_entry(file_id) for file_id in self.files]

    def set_drawing_type(self, file_id: str, drawing_type: str) -> Dict[str, Any]:
        """Apply a user-confirmed image type while preserving cached OCR/segmentation."""
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
            if isinstance(entry.parsed, dict):
                entry.parsed["drawing_type"] = drawing_type
            entry.evidence_pack = None
            entry.parse_error = None
            self.classification_warnings.pop(file_id, None)
            return self._manifest_entry(file_id)

    def retry_parse(self, file_id: str) -> None:
        """Clear cached parse state so the caller can retry after setup changes."""
        with self._lock:
            entry = self.files.get(file_id)
            if entry is None:
                raise KeyError(f"Unknown file_id: {file_id}")
            entry.parsed = None
            entry.ifc_index = None
            entry.generic_cad = None
            entry.rendered_pdf_pages.clear()
            entry.evidence_pack = None
            entry.parse_status = "not_parsed"
            entry.parse_error = None

    # ------------------------------------------------------------------ Tier 1 Specialists

    def _answer_cad_generic_fallback(self, question: str, entry: FileEntry) -> Optional[Dict[str, Any]]:
        """Answer generic CAD layer/text/dimension lookups when semantic floor-plan rules abstain."""
        if entry.generic_cad is None:
            from cad_parser import parse_cad
            entry.generic_cad = parse_cad(entry.path)
        doc = entry.generic_cad
        q_cf = question.casefold()
        source = str(entry.path)

        if "layer" in q_cf and any(k in q_cf for k in ("how many", "count", "number of", "list", "which")):
            layers = [ly for ly in (doc.get("layers") or []) if int(ly.get("entity_count") or 0) > 0]
            if any(k in q_cf for k in ("how many", "count", "number of")) and "each" not in q_cf and "list" not in q_cf:
                return {
                    "status": "answered",
                    "value": len(layers),
                    "answer": len(layers),
                    "unit": "count",
                    "method": "cad_layer_inventory",
                    "evidence": [{"source": source, "layers": [ly["name"] for ly in layers[:30]]}],
                }
            if layers:
                summary = ", ".join(f"{ly['name']} ({ly['entity_count']})" for ly in layers[:25])
                return {
                    "status": "answered",
                    "answer": summary,
                    "method": "cad_layer_inventory",
                    "evidence": [{"source": source, "layers": [ly["name"] for ly in layers[:30]]}],
                }
        return None

    def _run_entry_specialist(
        self,
        question: str,
        entry: FileEntry,
        *,
        has_exact_twin: bool = False,
    ) -> Dict[str, Any]:
        """Run the deterministic format specialist on a single cached FileEntry."""
        with self._lock:
            try:
                if entry.format == "ifc":
                    from ifc_qa import IfcIndex, answer_ifc_question, answer_ifc_text_question
                    if entry.ifc_index is None:
                        entry.ifc_index = IfcIndex(str(entry.path))
                    entry.parse_status = "parsed"
                    entry.parse_error = None

                    if _is_ifc_text_query(question):
                        res = answer_ifc_text_question(question, entry.ifc_index)
                        if res.get("status") == "unsupported":
                            res = answer_ifc_question(question, entry.ifc_index)
                        return res

                    res = answer_ifc_question(question, entry.ifc_index)
                    if res.get("status") == "unsupported" and _is_ifc_area_query(question):
                        from bim_parser import parse_ifc
                        from bim_qa import answer_area_question
                        if entry.parsed is None:
                            entry.parsed = parse_ifc(entry.path)
                        area_res = answer_area_question(question, entry.parsed)
                        if area_res.get("status") == "answered" or res.get("status") == "unsupported":
                            res = area_res
                    if res.get("status") in {"unsupported", "not_found"}:
                        text_res = answer_ifc_text_question(question, entry.ifc_index)
                        if text_res.get("status") == "answered":
                            res = text_res
                    return res

                if entry.format in {"dxf", "dwg"}:
                    from cad_qa import answer_cad_question, build_floor_index
                    if entry.parsed is None:
                        entry.parsed = build_floor_index(entry.path)
                    entry.parse_status = "parsed"
                    entry.parse_error = None
                    res = answer_cad_question(question, entry.parsed)
                    if res.get("status") == "unsupported":
                        fallback = self._answer_cad_generic_fallback(question, entry)
                        if fallback is not None:
                            return fallback
                    return res

                if entry.format == "pdf":
                    from pdf_parser import answer_pdf_question, parse_pdf
                    if entry.parsed is None:
                        entry.parsed = parse_pdf(entry.path)
                    entry.parse_status = "parsed"
                    entry.parse_error = None
                    return answer_pdf_question(
                        question,
                        entry.parsed,
                        enable_visual_fallback=not has_exact_twin,
                        render_dir=self.render_dir,
                        model_path=self.count_model,
                        geometry_model_path=self.geometry_model,
                        cubicasa_repo_path=self.cubicasa_repo,
                        cubicasa_weights_path=self.cubicasa_weights,
                        object_detector_model=self.object_detector_model,
                    )

                if entry.format == "image":
                    from image_parser import answer_image_question, parse_image
                    needs_ocr = _image_question_needs_ocr(question)
                    if entry.parsed is None:
                        entry.parsed = parse_image(entry.path, run_ocr=needs_ocr)
                    elif needs_ocr and entry.parsed.get("ocr_status") == "skipped":
                        specialist_cached = entry.parsed.get("floorplan_specialist")
                        entry.parsed = parse_image(entry.path, run_ocr=True)
                        if specialist_cached is not None:
                            entry.parsed["floorplan_specialist"] = specialist_cached
                    entry.parsed["drawing_type"] = entry.drawing_type or "uncertain"
                    entry.parse_status = "parsed"
                    entry.parse_error = None
                    return answer_image_question(
                        question,
                        entry.parsed,
                        model_path=self.count_model,
                        geometry_model_path=self.geometry_model,
                        cubicasa_repo_path=self.cubicasa_repo,
                        cubicasa_weights_path=self.cubicasa_weights,
                        object_detector_model=self.object_detector_model,
                    )

                return {
                    "status": "unsupported",
                    "answer": None,
                    "reason": f"Unsupported file format: {entry.format}",
                    "evidence": [],
                }
            except Exception as exc:
                entry.parse_status = "error"
                entry.parse_error = f"{type(exc).__name__}: {exc}"
                return {
                    "status": "not_found",
                    "answer": None,
                    "reason": f"Could not parse {entry.path.name}: {type(exc).__name__}: {exc}",
                    "evidence": [],
                }

    # ------------------------------------------------------------------ Tier 2 LLM Reasoner

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
                    if entry.parsed is None or entry.parsed.get("ocr_status") == "skipped":
                        specialist_cached = (entry.parsed or {}).get("floorplan_specialist")
                        entry.parsed = parse_image(entry.path, run_ocr=True)
                        if specialist_cached is not None:
                            entry.parsed["floorplan_specialist"] = specialist_cached
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

    # ------------------------------------------------------------------ Routing & Validation

    def _select_group_for_question(
        self,
        question: str,
        forced_file_id: Optional[str],
    ) -> Tuple[Optional[str], List[FileEntry], Optional[List[Dict[str, Any]]]]:
        """Choose the relevant drawing group (and ordered twin entries) or request clarification."""
        if forced_file_id is not None:
            forced = self.files.get(forced_file_id)
            if forced is None:
                return None, [], None
            canon = self._canonical_group(forced.group_id)
            siblings = [
                self.files[fid]
                for fid in self._group_members_with_twins(forced.group_id)
                if fid in self.files and fid != forced_file_id
            ]
            # Prefer exact BIM/CAD twins ahead of rendered PDF/raster views of the same drawing,
            # while keeping `forced` ahead of equal-or-lower priority formats.
            exact_twins = sorted(
                [s for s in siblings if _FORMAT_PRIORITY.get(s.format, 99) < _FORMAT_PRIORITY.get(forced.format, 99)],
                key=lambda item: _FORMAT_PRIORITY.get(item.format, 99),
            )
            fallback_siblings = sorted(
                [s for s in siblings if s not in exact_twins],
                key=lambda item: _FORMAT_PRIORITY.get(item.format, 99),
            )
            return canon, [*exact_twins, forced, *fallback_siblings], None

        canonical_groups = list(
            dict.fromkeys(self._canonical_group(e.group_id) for e in self.files.values())
        )
        if not canonical_groups:
            return None, [], None

        def _ordered_group_entries(canon_group: str) -> List[FileEntry]:
            members = [
                self.files[fid]
                for fid in self._group_members_with_twins(canon_group)
                if fid in self.files
            ]
            members.sort(key=lambda item: _FORMAT_PRIORITY.get(item.format, 99))
            return members

        if len(canonical_groups) == 1:
            g = canonical_groups[0]
            return g, _ordered_group_entries(g), None

        # Check explicit filename or stem mention in the question
        q_cf = question.casefold()
        mentioned_groups = [
            g for g in canonical_groups
            if any(
                e.path.name.casefold() in q_cf or (len(e.path.stem) >= 4 and e.path.stem.casefold() in q_cf)
                for e in _ordered_group_entries(g)
            )
        ]
        if len(mentioned_groups) == 1:
            g = mentioned_groups[0]
            return g, _ordered_group_entries(g), None

        # Format-specific disambiguation when only one group contains that format
        if any(k in q_cf for k in ("ifc", "in the model", "im modell", "dans le modèle", "dans le modele", "mudelis")):
            ifc_groups = [g for g in canonical_groups if any(e.format == "ifc" for e in _ordered_group_entries(g))]
            if len(ifc_groups) == 1:
                g = ifc_groups[0]
                return g, _ordered_group_entries(g), None
        if "schedule" in q_cf or "scheduled" in q_cf:
            pdf_groups = [g for g in canonical_groups if any(e.format == "pdf" for e in _ordered_group_entries(g))]
            if len(pdf_groups) == 1:
                g = pdf_groups[0]
                return g, _ordered_group_entries(g), None

        # Active conversation group continuity for follow-up questions
        if self._active_group in canonical_groups and self.history:
            if re.match(r"^(?:what about|how about|and\b|also\b|in that|on that|which one|what is its)\b", q_cf.strip()):
                return self._active_group, _ordered_group_entries(self._active_group), None

        options = []
        for g in canonical_groups:
            members = _ordered_group_entries(g)
            if members:
                rep = members[0]
                options.append({
                    "file_id": rep.file_id,
                    "name": rep.path.name,
                    "format": rep.format,
                    "related_files": [m.path.name for m in members[1:]],
                })
        return None, [], options

    def _candidate_entries(self, forced_file_id: Optional[str]) -> List[FileEntry]:
        """Select candidate files across drawing groups for Tier-2 multi-file LLM reasoning."""
        if forced_file_id is not None:
            _, ordered, _ = self._select_group_for_question("", forced_file_id)
            return ordered[:2]

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
            if unit in {"count", "m2", "ft2", "m", "m3"} and float(primary) < 0:
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

    def _decorate_entry_metadata(
        self,
        payload: Dict[str, Any],
        entry: Optional[FileEntry],
        query_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = self._validate_answer(payload)
        if entry is not None:
            out["file_id"] = entry.file_id
            out["file"] = entry.path.name
            out["format"] = entry.format
            out["drawing_type"] = entry.drawing_type
            siblings = self._group_members_with_twins(entry.group_id)
            out["related_files"] = [
                self.files[fid].path.name
                for fid in siblings
                if fid in self.files and fid != entry.file_id
            ]
            self._active_group = self._canonical_group(entry.group_id)
        out["query_interpretation"] = query_meta

        if out.get("status") == "answered":
            self.history.append({
                "question": question_val if (question_val := query_meta.get("question")) else "",
                "file_id": out.get("file_id"),
                "file": out.get("file"),
                "answer": out.get("value", out.get("answer")),
                "unit": out.get("unit"),
                "explanation": out.get("reason") or out.get("method"),
            })
            if len(self.history) > 12:
                self.history = self.history[-12:]
        return out

    def _answer_with_llm_reasoner(
        self,
        question: str,
        *,
        forced_file_id: Optional[str],
        candidate_entries: Sequence[FileEntry],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Run Tier-2 grounded LLM reasoning over EvidencePacks; return (result, error_str)."""
        reasoner = self.reasoner_fn or request_reasoning_decision
        entries = list(candidate_entries) if candidate_entries else self._candidate_entries(forced_file_id)
        packs: List[EvidencePack] = []
        last_parse_error: Optional[Tuple[FileEntry, Exception]] = None
        for entry in entries:
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
                return None, f"Could not parse {failed_entry.path.name}: {type(exc).__name__}: {exc}"
            return None, "No readable drawing files available for LLM reasoning."

        twin_map = {
            entry.file_id: [
                self.files[fid].path.name
                for fid in self._group_members_with_twins(entry.group_id)
                if fid in self.files and fid != entry.file_id
            ]
            for entry in entries
            if entry.file_id in {p.file_id for p in packs}
        }

        try:
            decision, provider = reasoner(
                question,
                packs,
                history=self.history,
                forced_file_id=forced_file_id,
                twin_relationships=twin_map,
            )

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
                            forced_file_id=forced_file_id,
                            twin_relationships=twin_map,
                        )
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

        packs_by_id = {p.file_id: p for p in packs}
        clarification_options = (
            [{"file_id": p.file_id, "name": p.file_name, "format": p.format} for p in packs]
            if len(packs) > 1
            else None
        )
        chosen_pack, raw_result = execute_reasoning_decision(
            decision,
            packs_by_id,
            provider=provider,
            fallback_options=clarification_options,
            question=question,
        )
        chosen_entry = self.files.get(chosen_pack.file_id) if chosen_pack else self.files.get(packs[0].file_id)
        decorated = self._decorate_entry_metadata(
            raw_result,
            chosen_entry,
            {
                "planner": f"llm_grounded_reasoner ({provider})",
                "status": "reasoned",
                "operation": decision.get("operation"),
                "selected_count": len(decision.get("selected_ids") or []),
                "question": question,
            },
        )
        return decorated, None

    def answer(self, question: str, *, file_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer a natural-language question with format-prioritized specialists + LLM reasoning."""
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

        has_llm = (self.reasoner_fn is not None) or llm_is_configured()
        bypass_tier1 = (self.reasoner_fn is not None) or (
            has_llm and _needs_compositional_llm(question, has_history=bool(self.history))
        )

        canon_group, group_entries, clarification_options = self._select_group_for_question(question, file_id)

        # Multiple unrelated drawings uploaded without an explicit match:
        # let Tier 2 disambiguate if configured, else request clarification with file options.
        if clarification_options is not None:
            if has_llm:
                llm_res, _ = self._answer_with_llm_reasoner(
                    question,
                    forced_file_id=file_id,
                    candidate_entries=self._candidate_entries(None),
                )
                if llm_res is not None:
                    return llm_res
            return {
                "status": "clarification_required",
                "answer": "Multiple unrelated drawings are loaded in this conversation. Please select a drawing or mention its file name.",
                "reason": "Question could apply to multiple uploaded drawings.",
                "options": clarification_options,
                "evidence": [],
            }

        # Direct Tier-2 execution for compositional queries or injected test reasoners
        if bypass_tier1:
            llm_res, llm_err = self._answer_with_llm_reasoner(
                question,
                forced_file_id=file_id,
                candidate_entries=group_entries[:2],
            )
            if llm_res is not None and "not partial OCR text labels" not in str(llm_res.get("reason") or ""):
                return llm_res
        else:
            llm_err = None

        # Tier-1 deterministic specialist execution across format-ranked twins (IFC > DXF/DWG > PDF > Image)
        has_exact_twin = any(e.format in {"ifc", "dxf", "dwg"} for e in group_entries)
        best_non_answer: Optional[Tuple[FileEntry, Dict[str, Any]]] = None
        hard_abstained = False

        for entry in group_entries:
            entry_has_exact_twin = has_exact_twin and entry.format not in {"ifc", "dxf", "dwg"}
            res = self._run_entry_specialist(question, entry, has_exact_twin=entry_has_exact_twin)
            status = res.get("status")
            if status in {"answered", "clarification_required"}:
                meta: Dict[str, Any] = {
                    "planner": f"specialist_{entry.format}",
                    "status": "deterministic",
                    "question": question,
                }
                if llm_err:
                    meta["llm_fallback_warning"] = llm_err
                return self._decorate_entry_metadata(res, entry, meta)

            reason_cf = str(res.get("reason") or "").casefold()
            if status == "not_found" and any(m in reason_cf for m in _HARD_ABSTENTION_MARKERS):
                hard_abstained = True
                best_non_answer = (entry, res)
                break

            if best_non_answer is None or (
                best_non_answer[1].get("status") == "unsupported" and status == "not_found"
            ):
                best_non_answer = (entry, res)

        # Tier-2 LLM compositional fallback when Tier 1 did not answer (and did not hit a safety abstention)
        if has_llm and not hard_abstained and not bypass_tier1:
            llm_res, llm_err = self._answer_with_llm_reasoner(
                question,
                forced_file_id=file_id,
                candidate_entries=group_entries[:2],
            )
            if llm_res is not None and llm_res.get("status") in {"answered", "clarification_required"}:
                return llm_res

        fallback_entry, fallback_res = best_non_answer or (
            group_entries[0] if group_entries else None,
            {"status": "not_found", "answer": None, "reason": "No matching evidence found.", "evidence": []},
        )
        meta = {
            "planner": f"specialist_{fallback_entry.format}" if fallback_entry else "none",
            "status": "abstained",
            "question": question,
        }
        if llm_err:
            meta["llm_fallback_warning"] = llm_err
        return self._decorate_entry_metadata(fallback_res, fallback_entry, meta)

    def close(self) -> None:
        """Release this conversation's in-memory file metadata and parse cache."""
        with self._lock:
            self.files.clear()
            self._groups.clear()
            self._active_group = None
            self.history.clear()
            self.classification_warnings.clear()
            self._vision_classifier = None
