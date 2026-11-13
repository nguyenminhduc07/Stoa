"""Train the shared floor-plan count CNN from image-level QA labels.

Example:
    python train_floorplan_model.py --dataset-root junction-dataset-preview \
        --questions junction-dataset-preview/questions.jsonl --output models/cubicasa_counts.pt

Question answers provide weak image-level supervision only; they do not teach
the model the object locations. Validation is split by image to prevent
paraphrased questions for the same drawing from crossing the split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from floorplan_model import (
    CATEGORY_NAMES,
    DEFAULT_IMAGE_SIZE,
    MODEL_VERSION,
    SharedFloorplanCounter,
    require_torch,
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def question_category(question: str) -> Optional[str]:
    """Map the challenge's English/Finnish count templates to model heads."""
    q = _normalize(question)
    if "window" in q or "ikkun" in q:
        return "windows"
    if "toilet bowl" in q or "wc seat" in q or "wc-istuin" in q:
        return "toilet_bowls"
    if "sink" in q or "wash basin" in q or "tiskiallas" in q or "pesu- ja" in q:
        return "sinks"
    if "shower" in q or "bathtub" in q or "suihku" in q or "kylpyamme" in q:
        return "showers_bathtubs"
    if "storey" in q or "floors" in q or "kerrosta" in q:
        return "storeys"
    if "door" in q or "ovea" in q:
        return "doors"
    if "bedroom" in q or "makuuhuone" in q:
        return "bedrooms"
    if "kitchen" in q or "keitti" in q:
        return "kitchens"
    if "bathroom" in q or "washroom" in q or "kylpy- tai pesuhuone" in q:
        return "bathrooms"
    if "separate wc room" in q or "erillist" in q:
        return "wc_rooms"
    if "sauna" in q:
        return "saunas"
    if "balcon" in q or "parvek" in q:
        return "balconies"
    return None


def load_image_labels(
    questions_file: str | Path,
    dataset_root: str | Path,
) -> Dict[Path, Dict[str, int]]:
    """Group supported numeric answers into one sparse label vector per image."""
    root = Path(dataset_root).resolve()
    labels: Dict[Path, Dict[str, int]] = {}
    with Path(questions_file).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            relative = Path(str(row.get("path", "")))
            if len(relative.parts) < 2 or relative.parts[1].casefold() != "cubicasa5k":
                continue
            if row.get("type") != "count" or str(row.get("unit", "")).casefold() != "count":
                continue
            category = question_category(str(row.get("question", "")))
            if category not in CATEGORY_NAMES:
                continue
            try:
                value = int(row["answer"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid count answer on questions.jsonl line {line_number}") from exc
            if value < 0 or str(row["answer"]).strip() not in {str(value), f"{value}.0"}:
                raise ValueError(f"Expected a nonnegative integer answer on line {line_number}")
            image_path = (root / relative).resolve()
            if root not in image_path.parents:
                raise ValueError(f"Question path escapes dataset root on line {line_number}")
            image_labels = labels.setdefault(image_path, {})
            previous = image_labels.get(category)
            if previous is not None and previous != value:
                raise ValueError(
                    f"Conflicting {category} labels for {image_path.name}: {previous} and {value}"
                )
            image_labels[category] = value

    missing = [str(path) for path in labels if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"QA data refers to missing image: {missing[0]}")
    return labels


def split_by_image(
    labels: Dict[Path, Dict[str, int]], validation_fraction: float = 0.2, seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """Deterministically split unique drawings, never individual QA rows."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    images = sorted(labels, key=str)
    random.Random(seed).shuffle(images)
    validation_size = max(1, round(len(images) * validation_fraction))
    return images[validation_size:], images[:validation_size]


def _load_tensor(path: Path, image_size: int):
    torch = require_torch()
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Image model training requires Pillow: pip install pillow") from exc
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        offset = ((image_size - image.width) // 2, (image_size - image.height) // 2)
        canvas.paste(image, offset)
        pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        tensor = pixels.reshape(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
        return tensor.mul(2.0).sub(1.0)


class _FloorplanDataset:
    def __init__(self, image_paths: Iterable[Path], labels: Dict[Path, Dict[str, int]], image_size: int):
        self.image_paths = list(image_paths)
        self.labels = labels
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        torch = require_torch()
        path = self.image_paths[index]
        image = _load_tensor(path, self.image_size)
        # Quarter turns and reflections preserve all count labels and expose
        # the model to different drawing orientations.
        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, dims=(2,))
            if random.random() < 0.5:
                image = torch.flip(image, dims=(1,))
            image = torch.rot90(image, k=random.randrange(4), dims=(1, 2))
        target = torch.zeros(len(CATEGORY_NAMES), dtype=torch.float32)
        mask = torch.zeros(len(CATEGORY_NAMES), dtype=torch.bool)
        for category, value in self.labels[path].items():
            idx = CATEGORY_NAMES.index(category)
            target[idx] = math.log1p(value)
            mask[idx] = True
        return image, target, mask

    @property
    def augment(self):
        return False


class _TrainingDataset(_FloorplanDataset):
    @property
    def augment(self):
        return True


def _predict_matrix(model, images, torch):
    output = model(images)
    return torch.stack([output[name] for name in CATEGORY_NAMES], dim=1)


def train(
    questions_file: str | Path,
    dataset_root: str | Path,
    output_file: str | Path,
    *,
    epochs: int = 30,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    torch = require_torch()
    from torch.utils.data import DataLoader

    random.seed(seed)
    torch.manual_seed(seed)
    labels = load_image_labels(questions_file, dataset_root)
    train_paths, validation_paths = split_by_image(labels, seed=seed)
    train_loader = DataLoader(
        _TrainingDataset(train_paths, labels, image_size),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    validation_loader = DataLoader(
        _FloorplanDataset(validation_paths, labels, image_size),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SharedFloorplanCounter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    best_category_mae: Dict[str, Optional[float]] = {}

    for epoch in range(epochs):
        model.train()
        for images, targets, mask in train_loader:
            images, targets, mask = images.to(device), targets.to(device), mask.to(device)
            predicted = _predict_matrix(model, images, torch)
            if not mask.any():
                continue
            loss = ((predicted - targets).square() * mask).sum() / mask.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        total_labels = 0
        absolute_errors = torch.zeros(len(CATEGORY_NAMES), device=device)
        category_counts = torch.zeros(len(CATEGORY_NAMES), device=device)
        with torch.no_grad():
            for images, targets, mask in validation_loader:
                images, targets, mask = images.to(device), targets.to(device), mask.to(device)
                predicted = _predict_matrix(model, images, torch)
                if mask.any():
                    total_loss += ((predicted - targets).square() * mask).sum().item()
                    total_labels += int(mask.sum().item())
                predicted_counts = torch.expm1(predicted.clamp_min(0))
                target_counts = torch.expm1(targets)
                absolute_errors += ((predicted_counts - target_counts).abs() * mask).sum(dim=0)
                category_counts += mask.sum(dim=0)
        validation_loss = total_loss / max(1, total_labels)
        category_mae = {
            name: (float(absolute_errors[i] / category_counts[i]) if category_counts[i] else None)
            for i, name in enumerate(CATEGORY_NAMES)
        }
        print(f"epoch {epoch + 1}/{epochs}: val_log_mse={validation_loss:.4f}", flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_category_mae = category_mae

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint; no labeled validation examples were available.")
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_version": MODEL_VERSION,
        "categories": list(CATEGORY_NAMES),
        "image_size": image_size,
        "state_dict": best_state,
        "validation_log_mse": best_loss,
        "validation_mae_counts": best_category_mae,
        "train_image_count": len(train_paths),
        "validation_image_count": len(validation_paths),
    }, target)
    return {
        "checkpoint": str(target.resolve()),
        "train_image_count": len(train_paths),
        "validation_image_count": len(validation_paths),
        "validation_log_mse": best_loss,
        "validation_mae_counts": best_category_mae,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Root containing data/cubicasa5k")
    parser.add_argument("--questions", required=True, help="Challenge questions.jsonl")
    parser.add_argument("--output", default="models/cubicasa_count_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summarize", action="store_true", help="Show supported label counts and exit")
    args = parser.parse_args()

    labels = load_image_labels(args.questions, args.dataset_root)
    coverage = Counter(category for per_image in labels.values() for category in per_image)
    if args.summarize:
        print(f"images with labels: {len(labels)}")
        for category in CATEGORY_NAMES:
            print(f"{category}: {coverage[category]} image labels")
        return

    result = train(
        args.questions, args.dataset_root, args.output,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
