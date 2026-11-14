"""Rasterize CubiCasa model.svg annotations to room/icon label masks.

This adapter uses CubiCasa5K's own SVG loader so its class IDs match the
training and inference code in this project. It can process drawings in
parallel and skips masks that already exist, so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np

from floorplan_geometry_model import ICON_CLASSES, ROOM_CLASSES


Job = Tuple[str, str, str, str]


def _is_valid_mask(path: Path) -> bool:
    """Only skip complete masks with the arrays expected by the trainer."""
    try:
        with np.load(path, allow_pickle=False) as data:
            if not {"rooms", "icons"}.issubset(data.files):
                return False
            rooms, icons = data["rooms"], data["icons"]
            return rooms.ndim == 2 and icons.ndim == 2 and rooms.shape == icons.shape
    except Exception:
        return False


def _convert_one(job: Job) -> Tuple[str, str]:
    repo_text, dataset_text, output_text, svg_text = job
    repo, dataset, output, svg_path = map(Path, (repo_text, dataset_text, output_text, svg_text))
    try:
        repo_string = str(repo)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        # Imported in each worker process, then cached for its subsequent files.
        from floortrans.loaders.house import House
        from PIL import Image

        image_path = svg_path.parent / "F1_original.png"
        if not image_path.is_file():
            image_path = svg_path.parent / "F1_scaled.png"
        if not image_path.is_file():
            return "failed", f"no F1_original.png or F1_scaled.png beside {svg_path}"

        with Image.open(image_path) as image:
            width, height = image.size
        label = np.asarray(House(str(svg_path), height, width).get_segmentation_tensor())
        if label.shape != (2, height, width):
            raise ValueError(f"unexpected mask shape {label.shape}")
        rooms, icons = label[0].astype(np.uint8), label[1].astype(np.uint8)
        if rooms.max(initial=0) >= len(ROOM_CLASSES) or icons.max(initial=0) >= len(ICON_CLASSES):
            raise ValueError("annotation class IDs exceed expected CubiCasa class mapping")

        relative = image_path.relative_to(dataset)
        target = (output / relative).with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically so an interrupted conversion won't leave a partial NPZ
        # that a resumed run mistakes for a completed mask.
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, rooms=rooms, icons=icons)
        temporary.replace(target)
        return "converted", str(target)
    except Exception as exc:
        return "failed", f"{svg_path}: {type(exc).__name__}: {exc}"


def prepare_masks(
    cubicasa_repo: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 1,
    overwrite: bool = False,
) -> int:
    repo = Path(cubicasa_repo).resolve()
    dataset = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve()
    loader_file = repo / "floortrans" / "loaders" / "house.py"
    if not loader_file.is_file():
        raise FileNotFoundError(f"Expected upstream CubiCasa loader at {loader_file}")
    if not dataset.is_dir():
        raise FileNotFoundError(f"CubiCasa dataset directory not found: {dataset}")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    jobs: List[Job] = []
    failures = []
    skipped_existing = 0
    for svg_path in sorted(dataset.rglob("model.svg")):
        image_path = svg_path.parent / "F1_original.png"
        if not image_path.is_file():
            image_path = svg_path.parent / "F1_scaled.png"
        if not image_path.is_file():
            failures.append(f"no F1_original.png or F1_scaled.png beside {svg_path}")
            continue
        target = (output / image_path.relative_to(dataset)).with_suffix(".npz")
        if target.is_file() and not overwrite and _is_valid_mask(target):
            skipped_existing += 1
            continue
        jobs.append((str(repo), str(dataset), str(output), str(svg_path)))

    print(
        f"Preparing {len(jobs)} masks with {workers} worker(s); "
        f"skipping {skipped_existing} existing masks.",
        flush=True,
    )
    converted = 0
    completed = 0

    def collect(result: Tuple[str, str]) -> None:
        nonlocal converted, completed
        status, detail = result
        completed += 1
        if status == "converted":
            converted += 1
        else:
            failures.append(detail)
        if completed % 25 == 0 or completed == len(jobs):
            print(f"Processed {completed}/{len(jobs)}; converted {converted}; failed {len(failures)}", flush=True)

    if workers == 1:
        for job in jobs:
            collect(_convert_one(job))
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_convert_one, job) for job in jobs]
            for future in as_completed(futures):
                try:
                    collect(future.result())
                except Exception as exc:
                    failures.append(f"Worker failed: {type(exc).__name__}: {exc}")
                    completed += 1
                    if completed % 25 == 0 or completed == len(jobs):
                        print(f"Processed {completed}/{len(jobs)}; converted {converted}; failed {len(failures)}", flush=True)

    print(
        f"Converted {converted} SVG annotations into masks under {output}; "
        f"skipped {skipped_existing} existing masks.",
        flush=True,
    )
    if failures:
        print(f"Skipped/failed: {len(failures)}", flush=True)
        for message in failures[:20]:
            print(f"  {message}", flush=True)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cubicasa-repo", required=True, help="Local checkout of github.com/CubiCasa/CubiCasa5k")
    parser.add_argument("--dataset", required=True, help="Extracted upstream data/cubicasa5k directory")
    parser.add_argument("--output", default="data/cubicasa5k_masks", help="Output mask root")
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="Parallel SVG conversion workers (default: up to 8)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate masks even when output NPZ files exist")
    args = parser.parse_args()
    prepare_masks(args.cubicasa_repo, args.dataset, args.output, workers=args.workers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
