"""Evaluate IFC area question answering against the challenge JSONL answers."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from bim_parser import parse_ifc
from bim_qa import answer_area_question, score_numeric_answer


def _find_model(question_path: str, data_dir: Path, available: Dict[str, Path]) -> Optional[Path]:
    source = Path(question_path)
    direct = data_dir / source.relative_to("data")
    if direct.suffix.lower() == ".ifc" and direct.exists():
        return direct
    stem = source.stem
    if "_plan-" in stem:
        model_stem = stem.split("_plan-", 1)[0]
        candidate = available.get(model_stem)
        if candidate:
            return candidate
    return None


def evaluate(questions_path: Path, data_dir: Path) -> Dict[str, Any]:
    rows = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    area_questions = [
        row for row in rows
        if row.get("type") == "area"
        and ("/ifc_models/" in row.get("path", "").lower()
             or "/ifc_bench/" in row.get("path", "").lower())
    ]
    models_by_stem = {path.stem: path for path in data_dir.rglob("*.ifc")}
    parsed: Dict[Path, Dict[str, Any]] = {}
    counts = Counter()
    by_model = defaultdict(Counter)
    failures = []

    for question in area_questions:
        model_path = _find_model(question["path"], data_dir, models_by_stem)
        if model_path is None:
            counts["missing_model"] += 1
            by_model[question["path"]]["missing_model"] += 1
            failures.append({"path": question["path"], "question": question["question"],
                             "expected": question.get("answer"), "status": "missing_model"})
            continue
        if model_path not in parsed:
            parsed[model_path] = parse_ifc(model_path)
        result = answer_area_question(question["question"], parsed[model_path])
        counts[result["status"]] += 1
        scored = score_numeric_answer(result, question.get("answer"))
        counts["scored"] += int(scored is not None)
        counts["correct"] += int(scored is True)
        by_model[model_path.name]["total"] += 1
        by_model[model_path.name]["correct"] += int(scored is True)
        by_model[model_path.name]["not_found"] += int(result["status"] != "answered")
        if scored is not True:
            failures.append({
                "path": question["path"], "question": question["question"],
                "expected": question.get("answer"), "actual": result.get("value"),
                "status": result["status"], "method": result.get("method"),
            })

    return {
        "area_question_records": len(area_questions),
        "parsed_models": len(parsed),
        "counts": dict(counts),
        "by_model": {name: dict(c) for name, c in by_model.items()},
        "failures": failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IFC area answers against questions.jsonl")
    parser.add_argument("--questions", type=Path, default=Path("junction-dataset-preview/questions.jsonl"))
    parser.add_argument("--data", type=Path, default=Path("junction-dataset-preview/data"))
    parser.add_argument("--failures", type=int, default=20, help="Maximum failure examples to print")
    args = parser.parse_args()
    report = evaluate(args.questions, args.data)
    failures = report.pop("failures")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Failure examples:")
    print(json.dumps(failures[:args.failures], ensure_ascii=False, indent=2))
