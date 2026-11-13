"""Evaluate grounded PDF schedule answers against text-type challenge records."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from pdf_parser import answer_pdf_question, parse_pdf


def _normal(text: Any) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(char for char in value if char.isalnum())


def evaluate(questions_path: Path, data_dir: Path) -> Dict[str, Any]:
    rows = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [row for row in rows if row.get("type") == "text" and row.get("path", "").lower().endswith(".pdf")]
    cache: Dict[Path, Dict[str, Any]] = {}
    counts = Counter()
    failures = []
    for question in questions:
        path = data_dir / Path(question["path"]).relative_to("data")
        if path not in cache:
            cache[path] = parse_pdf(path)
        result = answer_pdf_question(question["question"], cache[path])
        counts[result["status"]] += 1
        matched = result.get("status") == "answered" and _normal(question.get("answer")) in _normal(result.get("answer"))
        counts["exact_normalized_match"] += int(matched)
        if not matched:
            failures.append({
                "path": question["path"], "question": question["question"],
                "expected": question.get("answer"), "actual": result.get("answer"),
                "status": result.get("status"),
            })
    return {
        "question_records": len(questions),
        "parsed_pdfs": len(cache),
        "counts": dict(counts),
        "failures": failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PDF schedule/text answers against questions.jsonl")
    parser.add_argument("--questions", type=Path, default=Path("junction-dataset-preview/questions.jsonl"))
    parser.add_argument("--data", type=Path, default=Path("junction-dataset-preview/data"))
    args = parser.parse_args()
    report = evaluate(args.questions, args.data)
    print(json.dumps(report, ensure_ascii=False, indent=2))
