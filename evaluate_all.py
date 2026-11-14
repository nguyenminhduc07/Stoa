"""One score for the whole system: route every question to its answerer and grade it.

Routing (most exact format first, per AGENTS.md):
  IFC model, or a plan PDF rendered from one  -> ifc_qa (counts, wall/slab quantities), bim_qa (room areas)
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
TEXT_STOPWORDS = {"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "including",
                  "is", "it", "of", "on", "or", "the", "their", "there", "these", "this", "to",
                  "was", "were", "what", "which", "with", "within", "used", "use", "building", "model"}


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


def _area_number(value: Any) -> Optional[float]:
    """Prefer an explicit summary total over floor or room-count numbers."""
    if isinstance(value, str):
        total = re.search(
            r"\btotal(?:\s+at\s+least)?\s*(?:area\s*)?[:=]?\s*"
            r"(-?\d[\d\s,]*(?:\.\d+)?)",
            value,
            re.IGNORECASE,
        )
        if total:
            return _number(total.group(1))
    return _number(value)


def grade(result: Dict[str, Any], row: Dict[str, Any]) -> bool:
    if result.get("status") != "answered":
        return False
    answer = result.get("value", result.get("answer"))
    expected, kind = row.get("answer"), row.get("type")
    if kind in NUMERIC_TYPES:
        actual = _area_number(answer) if kind == "area" else _number(answer)
        target = _number(expected)
        if actual is None or target is None:
            return False
        if kind == "count":
            return round(actual) == round(target)
        return abs(actual - target) <= max(abs(target) * 0.02, 0.001)
    return bool(_norm_text(expected)) and _norm_text(expected) in _norm_text(answer)


def _f1_overlap(expected: list, actual: list) -> float:
    from collections import Counter
    expected_counts, actual_counts = Counter(expected), Counter(actual)
    overlap = sum((expected_counts & actual_counts).values())
    if not expected_counts and not actual_counts:
        return 1.0
    if overlap == 0:
        return 0.0
    precision = overlap / sum(actual_counts.values())
    recall = overlap / sum(expected_counts.values())
    return 2 * precision * recall / (precision + recall)


def text_similarity(expected: Any, actual: Any) -> Dict[str, float]:
    """Return a lexical diagnostic for free-form text, not a correctness verdict."""
    expected_text, actual_text = str(expected or ""), str(actual or "")

    def words(text: str) -> list:
        value = unicodedata.normalize("NFKD", text.casefold())
        value = "".join(character for character in value if not unicodedata.combining(character))
        return [token for token in re.findall(r"[a-z0-9]+", value) if token not in TEXT_STOPWORDS]

    expected_tokens, actual_tokens = words(expected_text), words(actual_text)
    token_f1 = _f1_overlap(expected_tokens, actual_tokens)
    number_pattern = r"(?<![A-Za-z])[-+]?\d[\d, .]*(?:\.\d+)?%?"
    expected_numbers = [number for match in re.findall(number_pattern, expected_text)
                        if (number := _number(match)) is not None]
    actual_numbers = [number for match in re.findall(number_pattern, actual_text)
                      if (number := _number(match)) is not None]
    numeric_f1 = _f1_overlap([round(number, 3) for number in expected_numbers],
                             [round(number, 3) for number in actual_numbers])
    score = 0.7 * token_f1 + 0.3 * numeric_f1 if expected_numbers else token_f1
    return {"score": score, "token_f1": token_f1, "numeric_f1": numeric_f1}


# --------------------------------------------------------------------------- routing

class Router:
    """Picks an answerer per question and caches parsed documents."""

    def __init__(self, data_dir: Path, image_model: Optional[Path] = None,
                 cubicasa_repo: Optional[Path] = None, cubicasa_weights: Optional[Path] = None):
        self.data_dir = data_dir
        self.image_model = image_model
        self.cubicasa_repo = cubicasa_repo
        self.cubicasa_weights = cubicasa_weights
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
            from ifc_qa import IfcIndex, answer_ifc_question, answer_ifc_text_question
            index = self._cached("ifc-index", model, IfcIndex)
            if kind == "text":
                result = answer_ifc_text_question(question, index)
                if result.get("status") == "unsupported":
                    result = answer_ifc_question(question, index)
                return "ifc", result
            result = answer_ifc_question(question, index)
            if result.get("status") == "unsupported" and kind == "area":
                from bim_parser import parse_ifc
                from bim_qa import answer_area_question
                result = answer_area_question(question, self._cached("ifc", model, parse_ifc))
            if result.get("status") in {"unsupported", "not_found"}:
                text_result = answer_ifc_text_question(question, index)
                if text_result.get("status") == "answered":
                    result = text_result
            return "ifc", result

        dxf = path.with_suffix(".dxf")
        if dxf.exists():
            from cad_qa import answer_cad_question, build_floor_index
            return "dxf", answer_cad_question(question, self._cached("dxf", dxf, build_floor_index))
        dwg = path.with_suffix(".dwg")
        if dwg.exists():
            from cad_qa import answer_cad_question, build_floor_index
            return "dwg", answer_cad_question(question, self._cached("dwg", dwg, build_floor_index))

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
            needs_ocr = row.get("type") == "text" or any(
                term in question.casefold() for term in ("read", "label", "scale", "title", "drawing number")
            )
            document = self._cached("image", path, lambda p: parse_image(p, run_ocr=needs_ocr))
            return "image", answer_image_question(
                question, document, model_path=self.image_model,
                cubicasa_repo_path=self.cubicasa_repo,
                cubicasa_weights_path=self.cubicasa_weights,
            )
        return "other", {"status": "unsupported", "reason": f"No answerer for {suffix} files."}


# --------------------------------------------------------------------------- evaluation

def evaluate(questions_path: Path, data_dir: Path, *, image_model: Optional[Path] = None,
             cubicasa_repo: Optional[Path] = None, cubicasa_weights: Optional[Path] = None,
             sources: Optional[set] = None, holdout_only: bool = False, limit: Optional[int] = None,
             verbose: bool = False) -> Dict[str, Any]:
    rows = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sources:
        rows = [r for r in rows if Path(r["path"]).parts[1] in sources]
    if holdout_only:
        rows = _holdout_rows(rows, questions_path, data_dir)
    if limit:
        rows = rows[:limit]

    router = Router(data_dir, image_model, cubicasa_repo, cubicasa_weights)
    table: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    text_metrics_by_source: Dict[str, list] = defaultdict(list)
    routes, failures = Counter(), []
    numeric_total = numeric_correct = 0
    text_exact_matches = 0
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
        row_text_metrics = None
        if row.get("type") in NUMERIC_TYPES:
            numeric_total += 1
            numeric_correct += ok
        elif row.get("type") == "text":
            actual = result.get("value", result.get("answer"))
            row_text_metrics = text_similarity(row.get("answer"), actual)
            text_metrics_by_source[source].append(row_text_metrics)
            cell["text_similarity_sum"] += row_text_metrics["score"]
            cell["text_token_f1_sum"] += row_text_metrics["token_f1"]
            cell["text_numeric_f1_sum"] += row_text_metrics["numeric_f1"]
            cell["text_scored"] += 1
            text_exact_matches += ok
        routes[f"{source} -> {route}"] += 1
        if not ok:
            failure = {"path": row["path"], "type": row.get("type"), "question": row["question"],
                             "expected": row.get("answer"), "status": result.get("status"),
                             "actual": result.get("value", result.get("answer")),
                             "reason": result.get("reason") or result.get("method")}
            if row_text_metrics is not None:
                failure["text_similarity"] = row_text_metrics
            failures.append(failure)
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
        "strict_exact_correct": totals["correct"],
        "numeric_questions": numeric_total,
        "numeric_correct": numeric_correct,
        "numeric_accuracy": round(numeric_correct / max(1, numeric_total), 4),
        "text_questions": sum(len(items) for items in text_metrics_by_source.values()),
        "text_exact_matches": text_exact_matches,
        "text_quality_by_source": {
            source: {
                "count": len(items),
                "mean_overlap_f1": round(sum(item["score"] for item in items) / len(items), 4),
                "mean_token_f1": round(sum(item["token_f1"] for item in items) / len(items), 4),
                "mean_numeric_f1": round(sum(item["numeric_f1"] for item in items) / len(items), 4),
            }
            for source, items in sorted(text_metrics_by_source.items())
        },
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
    print(f"\nStrict exact/tolerance score: {report['strict_exact_correct']}/{report['questions']} "
          f"({100 * report['accuracy']:.1f}%), {report['answered']} answered, {report['seconds']}s")
    if report.get("numeric_questions"):
        print(f"Numeric accuracy: {report['numeric_correct']}/{report['numeric_questions']} "
              f"({100 * report['numeric_accuracy']:.1f}%)")
    if report.get("text_questions"):
        exact_rate = 100 * report["text_exact_matches"] / report["text_questions"]
        print(f"Free-text exact matches: {report['text_exact_matches']}/{report['text_questions']} ({exact_rate:.1f}%)")
        for source, quality in report["text_quality_by_source"].items():
            print(f"  {source}: lexical/numeric overlap F1 {quality['mean_overlap_f1']:.3f} "
                  f"(token {quality['mean_token_f1']:.3f}, numbers {quality['mean_numeric_f1']:.3f})")
        print("  Text overlap is a diagnostic proxy, not a semantic correctness verdict.\n")
    else:
        print()
    print(f"{'source/type':38} {'strict':>8} {'answered':>9} {'total':>7}   score")
    for key, cell in report["by_source_type"].items():
        if cell.get("text_scored"):
            score = cell["text_similarity_sum"] / cell["text_scored"]
            label = f"F1 {score:0.3f}"
        else:
            score = 100 * cell.get("correct", 0) / max(1, cell["total"])
            label = f"{score:5.1f}%"
        print(f"{key:38} {cell.get('correct', 0):>8} {cell.get('answered', 0):>9} {cell['total']:>7}  {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route every question to its answerer and score the whole system")
    parser.add_argument("--questions", type=Path, default=Path("junction-dataset-preview/questions.jsonl"))
    parser.add_argument("--data", type=Path, default=Path("junction-dataset-preview/data"))
    parser.add_argument("--image-model", type=Path, help="Optional legacy project count-model checkpoint")
    parser.add_argument("--cubicasa-repo", type=Path, help="Official CubiCasa5K checkout for pretrained raster counts")
    parser.add_argument("--cubicasa-weights", type=Path, help="Official model_best_val_loss_var.pkl checkpoint")
    parser.add_argument("--holdout-only", action="store_true",
                        help="Score image questions only on the CNN's validation drawings (avoid train leakage)")
    parser.add_argument("--sources", nargs="*", help="Only these data/<source> folders, e.g. swiss_dwellings ifc_models")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--failures-out", type=Path, help="Write every failed question to this JSONL file")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    report = evaluate(args.questions, args.data, image_model=args.image_model,
                      cubicasa_repo=args.cubicasa_repo, cubicasa_weights=args.cubicasa_weights,
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
