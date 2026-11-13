"""Evaluate DXF floor-plan answers (cad_qa.py) against questions.jsonl.

Uses the challenge scoring: counts exact, areas/lengths/volumes within 2%.
Questions are matched to a DXF with the same stem as the question's file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from cad_qa import answer_cad_question, build_floor_index, find_cad_twin


def score(result: Dict[str, Any], expected: Any, kind: str) -> bool:
    if result.get("status") != "answered" or not isinstance(expected, (int, float)):
        return False
    actual = float(result["value"])
    if kind == "count":
        return round(actual) == round(float(expected))
    return abs(actual - float(expected)) <= max(abs(float(expected)) * 0.02, 0.001)


def evaluate(questions_path: Path, data_dir: Path) -> Dict[str, Any]:
    rows = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cache: Dict[Path, Dict[str, Any]] = {}
    totals, by_kind, failures = Counter(), defaultdict(Counter), []
    for row in rows:
        if row.get("type") not in {"count", "area", "length", "volume"}:
            continue
        dxf = find_cad_twin(row["path"], data_dir)
        if dxf is None:
            continue
        if dxf not in cache:
            cache[dxf] = build_floor_index(dxf)
        result = answer_cad_question(row["question"], cache[dxf])
        ok = score(result, row["answer"], row["type"])
        totals["questions"] += 1
        totals["correct"] += ok
        totals[result["status"]] += 1
        method = result.get("method") or result.get("reason", "")
        by_kind[row["type"]]["total"] += 1
        by_kind[row["type"]]["correct"] += ok
        if not ok:
            failures.append({"path": row["path"], "question": row["question"], "expected": row["answer"],
                             "actual": result.get("value"), "status": result["status"], "method": method})
    return {"drawings": len(cache), "totals": dict(totals),
            "by_type": {k: dict(v) for k, v in by_kind.items()}, "failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DXF floor-plan answers against questions.jsonl")
    parser.add_argument("--questions", type=Path, default=Path("junction-dataset-preview/questions.jsonl"))
    parser.add_argument("--data", type=Path, default=Path("junction-dataset-preview/data"))
    parser.add_argument("--failures", type=int, default=20)
    args = parser.parse_args()
    report = evaluate(args.questions, args.data)
    failures = report.pop("failures")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Failure examples ({len(failures)} total):")
    print(json.dumps(failures[:args.failures], ensure_ascii=False, indent=2))
