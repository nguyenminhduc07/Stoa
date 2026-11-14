"""Train room/icon segmentation with optional QA-count auxiliary supervision.

The data directory must mirror the image directory and contain one NPZ per
image, with uint8 ``rooms`` and ``icons`` class-index masks. The CubiCasa
official loader can create these from each image's ``model.svg`` annotation.
QA counts are optional and are used only as a weak auxiliary loss.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from floorplan_geometry_model import (
    GEOMETRY_MODEL_VERSION,
    ICON_CLASSES,
    ROOM_CLASSES,
    FloorplanGeometryNet,
)
from floorplan_model import CATEGORY_NAMES, DEFAULT_IMAGE_SIZE, require_torch


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _qa_counts(questions_file: Optional[str | Path]) -> Dict[str, Dict[str, int]]:
    if not questions_file:
        return {}
    from train_floorplan_model import question_category

    answers: Dict[str, Dict[str, int]] = {}
    with Path(questions_file).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
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
                count = int(row["answer"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Bad image count on questions.jsonl line {line_number}") from exc
            key = relative.stem.casefold()
            prior = answers.setdefault(key, {}).get(category)
            if prior is not None and prior != count:
                raise ValueError(f"Conflicting QA counts for {key}: {prior} vs {count}")
            answers[key][category] = count
    return answers


def _qa_key_candidates(image_path: Path, images_root: Path) -> List[str]:
    relative = image_path.relative_to(images_root)
    candidates = [relative.stem.casefold()]
    # Upstream CubiCasa paths use <style>/<drawing-id>/F1_original.png;
    # challenge QA images use the equivalent <style>-<drawing-id>.png name.
    if len(relative.parts) >= 3:
        candidates.append(f"{relative.parts[-3]}-{relative.parts[-2]}".casefold())
    return candidates


def _find_mask_path(image_path: Path, images_root: Path, masks_root: Path) -> Path:
    relative = image_path.relative_to(images_root)
    return (masks_root / relative).with_suffix(".npz")


def _letterbox(image_path: Path, mask_path: Path, image_size: int):
    torch = require_torch()
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Geometry model training requires Pillow: pip install pillow") from exc

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        original_size = image.size
        scale = min(image_size / image.width, image_size / image.height)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        offset = ((image_size - new_size[0]) // 2, (image_size - new_size[1]) // 2)
        canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        canvas.paste(image, offset)
        image_tensor = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        image_tensor = image_tensor.reshape(image_size, image_size, 3).permute(2, 0, 1).float().div(127.5).sub(1.0)

    with np.load(mask_path, allow_pickle=False) as data:
        if not {"rooms", "icons"}.issubset(data.files):
            raise ValueError(f"Mask file must contain rooms and icons arrays: {mask_path}")
        room_mask = np.asarray(data["rooms"], dtype=np.uint8)
        icon_mask = np.asarray(data["icons"], dtype=np.uint8)
    if room_mask.ndim != 2 or icon_mask.ndim != 2 or room_mask.shape != icon_mask.shape:
        raise ValueError(f"Expected same-size 2D room/icon masks in {mask_path}")
    bad_room = ((room_mask >= len(ROOM_CLASSES)) & (room_mask != 255)).any()
    bad_icon = ((icon_mask >= len(ICON_CLASSES)) & (icon_mask != 255)).any()
    if bad_room or bad_icon:
        raise ValueError(f"Mask class ids do not match the documented CubiCasa class order: {mask_path}")
    if room_mask.shape != (original_size[1], original_size[0]):
        raise ValueError(
            f"Mask dimensions {icon_mask.shape[::-1]} do not match image {original_size}: {mask_path}"
        )
    room_img = Image.fromarray(room_mask).resize(new_size, Image.Resampling.NEAREST)
    icon_img = Image.fromarray(icon_mask).resize(new_size, Image.Resampling.NEAREST)
    room_canvas = Image.new("L", (image_size, image_size), 0)
    icon_canvas = Image.new("L", (image_size, image_size), 0)
    room_canvas.paste(room_img, offset)
    icon_canvas.paste(icon_img, offset)
    room_tensor = torch.from_numpy(np.asarray(room_canvas, dtype=np.uint8).copy()).long()
    icon_tensor = torch.from_numpy(np.asarray(icon_canvas, dtype=np.uint8).copy()).long()
    return image_tensor, room_tensor, icon_tensor


class GeometryDataset:
    def __init__(
        self,
        image_paths: Iterable[Path],
        images_root: Path,
        masks_root: Path,
        qa: Dict[str, Dict[str, int]],
        image_size: int,
        augment: bool = False,
    ):
        self.image_paths = list(image_paths)
        self.images_root = images_root
        self.masks_root = masks_root
        self.qa = qa
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        torch = require_torch()
        path = self.image_paths[index]
        mask_path = _find_mask_path(path, self.images_root, self.masks_root)
        image, rooms, icons = _letterbox(path, mask_path, self.image_size)
        if self.augment:
            if random.random() < 0.5:
                image, rooms, icons = (torch.flip(value, dims=(-1,)) for value in (image, rooms, icons))
            if random.random() < 0.5:
                image, rooms, icons = (torch.flip(value, dims=(-2,)) for value in (image, rooms, icons))
            turns = random.randrange(4)
            image, rooms, icons = (torch.rot90(value, turns, dims=(-2, -1)) for value in (image, rooms, icons))

        targets = torch.zeros(len(CATEGORY_NAMES), dtype=torch.float32)
        known = torch.zeros(len(CATEGORY_NAMES), dtype=torch.bool)
        for key in _qa_key_candidates(path, self.images_root):
            for category, count in self.qa.get(key, {}).items():
                slot = CATEGORY_NAMES.index(category)
                targets[slot] = math.log1p(count)
                known[slot] = True
        return image, rooms, icons, targets, known


def _masked_segmentation_loss(logits, target, torch):
    valid = (target >= 0) & (target < logits.shape[1])
    if not valid.any():
        return logits.sum() * 0.0
    ce = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(target.clamp(0, logits.shape[1] - 1), logits.shape[1]).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probabilities = probabilities * valid_f
    one_hot = one_hot * valid_f
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = (1.0 - dice[present]).mean() if present.any() else logits.sum() * 0.0
    return ce + dice_loss


def _update_iou(confusion, prediction, target, class_count, torch):
    valid = (target >= 0) & (target < class_count)
    encoded = target[valid] * class_count + prediction[valid]
    confusion += torch.bincount(encoded, minlength=class_count * class_count).reshape(class_count, class_count)


def train_geometry_model(
    images_dir: str | Path,
    masks_dir: str | Path,
    output_file: str | Path,
    *,
    questions_file: Optional[str | Path] = None,
    epochs: int = 30,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = 2,
    seed: int = 42,
) -> Dict[str, object]:
    torch = require_torch()
    from torch.utils.data import DataLoader

    images_root = Path(images_dir).resolve()
    masks_root = Path(masks_dir).resolve()
    image_paths = sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No PNG/JPEG images found under {images_root}")
    image_paths = [p for p in image_paths if _find_mask_path(p, images_root, masks_root).is_file()]
    if not image_paths:
        raise FileNotFoundError(
            f"No matching NPZ masks under {masks_root}. Each image needs a same-relative-path .npz "
            "containing uint8 'rooms' and 'icons' class-index arrays."
        )

    qa = _qa_counts(questions_file)
    random.seed(seed)
    torch.manual_seed(seed)
    shuffled = list(image_paths)
    random.Random(seed).shuffle(shuffled)
    validation_n = max(1, round(len(shuffled) * 0.2))
    validation_paths, training_paths = shuffled[:validation_n], shuffled[validation_n:]
    train_loader = DataLoader(
        GeometryDataset(training_paths, images_root, masks_root, qa, image_size, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    validation_loader = DataLoader(
        GeometryDataset(validation_paths, images_root, masks_root, qa, image_size),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}; labeled drawings={len(image_paths)}; train={len(training_paths)}; validation={len(validation_paths)}", flush=True)
    model = FloorplanGeometryNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    best_metrics: Dict[str, object] = {}
    for epoch in range(epochs):
        model.train()
        for images, rooms, icons, counts, known in train_loader:
            images, rooms, icons = images.to(device), rooms.to(device), icons.to(device)
            counts, known = counts.to(device), known.to(device)
            prediction = model(images)
            room_loss = _masked_segmentation_loss(prediction["rooms"], rooms, torch)
            icon_loss = _masked_segmentation_loss(prediction["icons"], icons, torch)
            if known.any():
                count_loss = ((prediction["count_log1p"] - counts).square() * known).sum() / known.sum()
            else:
                count_loss = prediction["count_log1p"].sum() * 0.0
            loss = room_loss + icon_loss + 0.15 * count_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        loss_sum = 0.0
        valid_pixels = 0
        room_cm = torch.zeros((len(ROOM_CLASSES), len(ROOM_CLASSES)), dtype=torch.int64, device=device)
        icon_cm = torch.zeros((len(ICON_CLASSES), len(ICON_CLASSES)), dtype=torch.int64, device=device)
        count_abs = torch.zeros(len(CATEGORY_NAMES), device=device)
        count_n = torch.zeros(len(CATEGORY_NAMES), device=device)
        with torch.no_grad():
            for images, rooms, icons, counts, known in validation_loader:
                images, rooms, icons = images.to(device), rooms.to(device), icons.to(device)
                counts, known = counts.to(device), known.to(device)
                prediction = model(images)
                room_loss = _masked_segmentation_loss(prediction["rooms"], rooms, torch)
                icon_loss = _masked_segmentation_loss(prediction["icons"], icons, torch)
                if known.any():
                    count_loss = ((prediction["count_log1p"] - counts).square() * known).sum() / known.sum()
                    pred_count = torch.expm1(prediction["count_log1p"].clamp_min(0))
                    target_count = torch.expm1(counts)
                    count_abs += ((pred_count - target_count).abs() * known).sum(dim=0)
                    count_n += known.sum(dim=0)
                else:
                    count_loss = prediction["count_log1p"].sum() * 0.0
                batch_loss = room_loss + icon_loss + 0.15 * count_loss
                loss_sum += float(batch_loss.item())
                valid_pixels += 1
                _update_iou(room_cm, prediction["rooms"].argmax(1), rooms, len(ROOM_CLASSES), torch)
                _update_iou(icon_cm, prediction["icons"].argmax(1), icons, len(ICON_CLASSES), torch)

        val_loss = loss_sum / max(1, valid_pixels)
        def mean_iou(confusion):
            confusion = confusion.float()
            denom = confusion.sum(0) + confusion.sum(1) - confusion.diag()
            present = denom > 0
            return float((confusion.diag()[present] / denom[present]).mean()) if present.any() else 0.0

        metrics: Dict[str, object] = {
            "validation_loss": val_loss,
            "room_mean_iou": mean_iou(room_cm),
            "icon_mean_iou": mean_iou(icon_cm),
            "qa_count_mae": {
                name: (float(count_abs[i] / count_n[i]) if count_n[i] else None)
                for i, name in enumerate(CATEGORY_NAMES)
            },
        }
        print(
            f"epoch {epoch + 1}/{epochs}: val_loss={val_loss:.4f}, "
            f"room_mIoU={metrics['room_mean_iou']:.3f}, icon_mIoU={metrics['icon_mean_iou']:.3f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No geometry checkpoint was produced")
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_version": GEOMETRY_MODEL_VERSION,
        "rooms": list(ROOM_CLASSES),
        "icons": list(ICON_CLASSES),
        "count_categories": list(CATEGORY_NAMES),
        "image_size": image_size,
        "state_dict": best_state,
        "train_image_count": len(training_paths),
        "validation_image_count": len(validation_paths),
        "metrics": best_metrics,
    }, output)
    return {"checkpoint": str(output.resolve()), **best_metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Image root; may be the full CubiCasa image tree")
    parser.add_argument("--masks", required=True, help="Mirrored root of NPZ room/icon class masks")
    parser.add_argument("--questions", help="Optional challenge questions.jsonl for weak count supervision")
    parser.add_argument("--output", default="models/cubicasa_geometry.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_geometry_model(
        args.images, args.masks, args.output,
        questions_file=args.questions, epochs=args.epochs,
        image_size=args.image_size, batch_size=args.batch_size, seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
