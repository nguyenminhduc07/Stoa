"""Rasterize CubiCasa model.svg annotations to room/icon label masks.

This small adapter uses CubiCasa5K's own SVG loader so its class IDs match the
training and inference code in this project. Point --cubicasa-repo at a local
checkout of the upstream repository and --dataset at its extracted
data/cubicasa5k directory. The script writes only generated NPZ masks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from floorplan_geometry_model import ICON_CLASSES, ROOM_CLASSES


def prepare_masks(cubicasa_repo: str | Path, dataset_dir: str | Path, output_dir: str | Path) -> int:
    repo = Path(cubicasa_repo).resolve()
    dataset = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve()
    loader_file = repo / "floortrans" / "loaders" / "house.py"
    if not loader_file.is_file():
        raise FileNotFoundError(f"Expected upstream CubiCasa loader at {loader_file}")
    if not dataset.is_dir():
        raise FileNotFoundError(f"CubiCasa dataset directory not found: {dataset}")

    sys.path.insert(0, str(repo))
    try:
        from floortrans.loaders.house import House
    except ImportError as exc:
        raise RuntimeError(
            "The official CubiCasa SVG loader dependencies are missing. Install the upstream "
            "requirements, including numpy, scikit-image, and svgpathtools."
        ) from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Mask conversion needs Pillow: pip install pillow") from exc

    converted = 0
    failures = []
    for svg_path in sorted(dataset.rglob("model.svg")):
        image_path = svg_path.parent / "F1_original.png"
        if not image_path.is_file():
            image_path = svg_path.parent / "F1_scaled.png"
        if not image_path.is_file():
            failures.append(f"no F1_original.png or F1_scaled.png beside {svg_path}")
            continue

        try:
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
            np.savez_compressed(target, rooms=rooms, icons=icons)
            converted += 1
        except Exception as exc:
            failures.append(f"{svg_path}: {type(exc).__name__}: {exc}")

    print(f"Converted {converted} SVG annotations into masks under {output}")
    if failures:
        print(f"Skipped/failed: {len(failures)}")
        for message in failures[:20]:
            print(f"  {message}")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cubicasa-repo", required=True, help="Local checkout of github.com/CubiCasa/CubiCasa5k")
    parser.add_argument("--dataset", required=True, help="Extracted upstream data/cubicasa5k directory")
    parser.add_argument("--output", default="data/cubicasa5k_masks", help="Output mask root")
    args = parser.parse_args()
    prepare_masks(args.cubicasa_repo, args.dataset, args.output)


if __name__ == "__main__":
    main()
