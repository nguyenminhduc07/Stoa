# AGENTS.md — shared rules for every coding agent in this repo

Read this file fully before making changes. It applies to Claude Code, Codex, Cursor,
Gemini CLI and any other agent. Humans: keep it short and current.

## Goal
Stoa challenge (Junction 2026): answer natural-language questions about construction
drawings and BIM models **correctly and verifiably**. Every answer must carry evidence
(file + page/element id + bbox or GUID) so a human can check it in seconds.

Scoring (from the dataset README): counts must be exact; areas, lengths, volumes within 2%;
text answers are free-form. Questions come in English and the local language (fi, de, fr, et, da).

## Data
Dataset is NOT in the repo. Default location: `junction-dataset-preview/` next to the code
(`questions.jsonl` + `data/<source>/...`). Never commit dataset files, model weights or API keys.
Files with the same stem and a different extension are the same drawing; prefer the most exact
format: IFC > DXF > vector PDF > raster image.

## Layout and owners
| file | purpose | owner |
|---|---|---|
| `file_classifier.py` | route files by format/source | Duc |
| `bim_parser.py` / `bim_qa.py` | IFC parsing + answers | Duc |
| `evaluate_bim_area.py` | eval for IFC area questions | Duc |
| `cad_parser.py` | DXF/DWG parsing (Swiss dwellings) | Duc |
| `pdf_parser.py` / `evaluate_pdf_text.py` | PDF schedules + eval | Duc |
| `image_parser.py`, `floorplan_model.py`, `floorplan_geometry_model.py`, `train_floorplan*.py`, `prepare_cubicasa_masks.py` | raster evidence and floor-plan CNNs | Codex |
| `cad_qa.py` / `evaluate_cad.py` | DXF floor-plan answers (Swiss dwellings) + eval | Minh |
| `evaluate_all.py` | routes every question to its answerer, one overall score | Minh |
| `room_label_qa.py` | CubiCasa room-type counts from OCR of room labels | Minh |
| _(add yours here)_ | | |

Only edit files you own. If you must change someone else's file, keep the diff minimal and
say so in the PR description.

## Shared contracts — do not change without telling the team
- Parsers return JSON-serialisable dicts and keep provenance (`source`, ids, bboxes).
- Answer functions return:
  `{"status": "answered" | "not_found", "value" | "answer": ..., "unit": ..., "method": str, "evidence": [...]}`
- Prefer `not_found` over guessing. A wrong confident answer is worse than abstaining.
- Authored quantities first; geometry-derived values only as a fallback, and label them in `method`.

## Rules
1. **No overfitting to preview files.** No hard-coded sheet names, project names, file names or
   answers. The final test set is different data. Generic rules only.
2. **Measure every change.** Run `evaluate_all.py` (and the relevant `evaluate_*.py`) before and after; put both numbers
   in the commit message or PR, e.g. `IFC area: 550/580 -> 580/580`.
3. **Small branches, merge often.** Branch per task (`feat/...`, `fix/...`), PR into `master`,
   merge every 2–3 hours. `master` must always run.
4. Python 3.10+. Add new dependencies to `requirements.txt` in the same commit.
5. Keep Windows line endings as they are; do not reformat files you did not change.
6. Record non-obvious choices in `DECISIONS.md` (one line: date, decision, why).

## Run
```
pip install -r requirements.txt
python evaluate_all.py --questions junction-dataset-preview/questions.jsonl --data junction-dataset-preview/data --failures-out failures.jsonl
python evaluate_bim_area.py --questions junction-dataset-preview/questions.jsonl --data junction-dataset-preview/data
python evaluate_pdf_text.py --questions junction-dataset-preview/questions.jsonl --data junction-dataset-preview/data
python evaluate_cad.py --questions junction-dataset-preview/questions.jsonl --data junction-dataset-preview/data
```
