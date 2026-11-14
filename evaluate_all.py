"""One score for the whole system: route every question to its answerer and grade it.

Routing (most exact format first, per AGENTS.md):
  IFC model, or a plan PDF rendered from one  -> bim_parser + bim_qa   (area questions)
  DXF twin with the same stem                  -> cad_qa
  PDF                                          -> pdf_parser
  raster image, room-type question             -> room_label_qa (OCR of room labels)
  raster image, symbol count                   -> image_parser (needs --image-model)

Grading follows the dataset README: counts exact, area/length/volume within 2%,
text by normalised containment (same rule as evaluate_pdf_text.py).

Example:
    python evaluate_all.py --questions junction-dataset-preview/questions.jsonl \
        --data junction-dataset-preview/data --failures-out failures.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

NUMERIC_TYPES = {"count", "area", "length", "volume"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------- grading

def _norm_text(text: Any) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(ch for ch in value if ch.isalnum())


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d[\d\s,]*(?:\.\d+)?", str(value))
    if not match:
        return None
    raw = match.group(0).replace(" ", "")
    raw = raw.replace(",", "") if "." in raw or raw.count(",") > 1 or re.search(r",\d{3}\b", raw) else raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def grade(result: Dict[str, Any], row: Dict[str, Any]) -> bool:
    if result.get("status") != "answered":
        return False
    answer = result.get("value", result.get("answer"))
    expected, kind = row.get("answer"), row.get("type")
    if kind in NUMERIC_TYPES:
        actual, target = _number(answer), _number(expected)
        if actual is None or target is None:
            return False
        if kind == "count":
            return round(actual) == round(target)
        return abs(actual - target) <= max(abs(target) * 0.02, 0.001)
    return bool(_norm_text(expected)) and _norm_text(expected) in _norm_text(answer)


# --------------------------------------------------------------------------- routing

class Router:
    """Picks an answerer per question and caches parsed documents."""

    def __init__(self, data_dir: Path, image_model: Optional[Path] = None):
        self.data_dir = data_dir
        self.image_model = image_model
        self.cache: Dict[Tuple[str, Path], Any] = {}
        self.ifc_by_stem = {p.stem: p for p in data_dir.rglob("*.ifc")}

    def _file(self, question_path: str) -> Path:
        path = Path(question_path)
        rel = path.relative_to("data") if path.parts and path.parts[0] == "data" else path
        return self.data_dir / rel

    def _cached(self, kind: str, path: Path, loader: Callable[[Path], Any]) -> Any:
        key = (kind, path)
        if key not in self.cache:
            self.cache[key] = loader(path)
        return self.cache[key]

    def _ifc_model(self, path: Path) -> Optional[Path]:
        if path.suffix.lower() == ".ifc":
            return path if path.exists() else None
        if "_plan-" in path.stem:  # plan PDFs rendered from <model>.ifc
            return self.ifc_by_stem.get(path.stem.split("_plan-", 1)[0])
        return None

    def route(self, row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        path = self._file(row["path"])
        question, kind = row["question"], row.get("type")

        model = self._ifc_model(path)
        if model is not None:
            if kind != "area":
                return "ifc", {"status": "unsupported", "reason": "IFC answerer handles area questions only."}
            from bim_parser import parse_ifc
            from bim_qa import answer_area_question
            return "ifc", answer_area_question(question, self._cached("ifc", model, parse_ifc))

        dxf = path.with_suffix(".dxf")
        if dxf.exists():
            from cad_qa import answer_cad_question, build_floor_index
            return "dxf", answer_cad_question(question, self._cached("dxf", dxf, build_floor_index))

        if not path.exists():
            return "missing", {"status": "missing_file", "reason": f"{path} not found"}

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pdf_parser import answer_pdf_question, parse_pdf
            return "pdf", answer_pdf_question(question, self._cached("pdf", path, parse_pdf))
        if suffix in IMAGE_EXTENSIONS:
            from room_label_qa import SUPPORTED, answer_room_question
            from train_floorplan_model import question_category
            category = question_category(question)
            if category in SUPPORTED:
                return "image-ocr", answer_room_question(question, path, category)
            from image_parser import answer_image_question, parse_image
            document = self._cached("image", path, lambda p: parse_image(p, run_ocr=False))
            return "image", answer_image_question(question, document, model_path=self.image_model)
        return "other", {"status": "unsupported", "reason": f"No answerer for {suffix} files."}


# --------------------------------------------------------------------------- evaluation

def evaluate(questions_path: Path, data_dir: Path, *, image_model: Optional[Path] = None,
             sources: Optional[set] = None, holdout_only: bool = False, limit: Optional[int] = None,
             verbose: bool = False) -> Dict[str, Any]:
    rows = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sources:
        rows = [r for r in rows if Path(r["path"]).parts[1] in sources]
    if holdout_only:
        rows = _holdout_rows(rows, questions_path, data_dir)
    if limit:
        rows = rows[:limit]

    router = Router(data_dir, image_model)
    table: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    routes, failures = Counter(), []
    started = time.time()
    for i, row in enumerate(rows, 1):
        source = Path(row["path"]).parts[1] if len(Path(row["path"]).parts) > 1 else "?"
        try:
            route, result = router.route(row)
        except Exception as exc:  # one broken drawing must not stop the whole run
            route, result = "error", {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            if verbose:
                traceback.print_exc()
        ok = grade(result, row)
        cell = table[(source, row.get("type", "?"))]
        cell["total"] += 1
        cell["correct"] += ok
        cell["answered"] += result.get("status") == "answered"
        routes[f"{source} -> {route}"] += 1
        if not ok:
            failures.append({"path": row["path"], "type": row.get("type"), "question": row["question"],
                             "expected": row.get("answer"), "status": result.get("status"),
                             "actual": result.get("value", result.get("answer")),
                             "reason": result.get("reason") or result.get("method")})
        if verbose and i % 500 == 0:
            print(f"  {i}/{len(rows)} questions, {time.time() - started:.0f}s", flush=True)

    totals = Counter()
    for cell in table.values():
        totals.update(cell)
    return {
        "questions": len(rows),
        "correct": totals["correct"],
        "answered": totals["answered"],
        "accuracy": round(totals["correct"] / max(1, len(rows)), 4),
        "seconds": round(time.time() - started, 1),
        "by_source_type": {f"{s}/{t}": dict(c) for (s, t), c in sorted(table.items())},
        "routes": dict(routes),
        "failures": failures,
    }


def _holdout_rows(rows, questions_path: Path, data_dir: Path):
    """Keep image questions only for drawings in the CNN's validation split."""
    from train_floorplan_model import load_image_labels, split_by_image
    _, validation = split_by_image(load_image_labels(questions_path, data_dir.parent))
    keep = {Path(p).resolve() for p in validation}
    out = []
    for row in rows:
        path = (data_dir / Path(row["path"]).relative_to("data")).resolve()
        if path.suffix.lower() not in IMAGE_EXTENSIONS or path in keep:
            out.append(row)
    return out


def print_report(report: Dict[str, Any]) -> None:
    print(f"\nOverall: {report['correct']}/{report['questions']} correct "
          f"({100 * report['accuracy']:.1f}%), {report['answered']} answered, {report['seconds']}s\n")
    print(f"{'source/type':38} {'correct':>8} {'answered':>9} {'total':>7}   acc")
    for key, cell in report["by_source_type"].items():
        acc = 100 * cell.get("correct", 0) / max(1, cell["total"])
        print(f"{key:38} {cell.get('correct', 0):>8} {cell.get('answered', 0):>9} {cell['total']:>7}  {acc:5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route every question to its answerer and score the whole system")
    parser.add_argument("--questions", type=Path, default=Path("junction-dataset-preview/questions.jsonl"))
    parser.add_argument("--data", type=Path, default=Path("junction-dataset-preview/data"))
    parser.add_argument("--image-model", type=Path, help="Floor-plan CNN checkpoint for image count questions")
    parser.add_argument("--holdout-only", action="store_true",
                        help="Score image questions only on the CNN's validation drawings (avoid train leakage)")
    parser.add_argument("--sources", nargs="*", help="Only these data/<source> folders, e.g. swiss_dwellings ifc_models")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--failures-out", type=Path, help="Write every failed question to this JSONL file")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    report = evaluate(args.questions, args.data, image_model=args.image_model,
                      sources=set(args.sources) if args.sources else None,
                      holdout_only=args.holdout_only, limit=args.limit, verbose=args.verbose)
    failures = report.pop("failures")
    if args.failures_out:
        with args.failures_out.open("w", encoding="utf-8") as out:
            for item in failures:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
        print("\nRoutes:", json.dumps(report["routes"], ensure_ascii=False))
