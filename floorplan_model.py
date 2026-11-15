"""Shared CNN with task-specific count heads for residential floor plans.

The QA pairs supervise image-level counts. They do not identify individual
objects, so this model predicts counts but does not localize them.
"""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CATEGORY_NAMES = (
    "windows",
    "toilet_bowls",
    "sinks",
    "showers_bathtubs",
    "storeys",
    "doors",
    "bedrooms",
    "kitchens",
    "bathrooms",
    "wc_rooms",
    "saunas",
    "balconies",
)
MODEL_VERSION = 1
DEFAULT_IMAGE_SIZE = 512


try:
    import torch
    from torch import nn
except ImportError:  # Keep metadata/image parsing usable without ML packages.
    torch = None
    nn = None


if nn is not None:
    class _ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            )
            self.skip = (
                nn.Identity() if in_channels == out_channels and stride == 1 else
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

        def forward(self, x):
            return self.layers(x) + self.skip(x)


    class SharedFloorplanCounter(nn.Module):
        """Shared convolutional encoder and one scalar regression head per count."""

        def __init__(self, categories: Iterable[str] = CATEGORY_NAMES):
            super().__init__()
            self.categories = tuple(categories)
            channels = (32, 64, 128, 192)
            blocks = []
            in_channels = 3
            for out_channels in channels:
                blocks.append(_ConvBlock(in_channels, out_channels, stride=2))
                in_channels = out_channels
            self.encoder = nn.Sequential(*blocks)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            feature_size = channels[-1] * 2
            self.shared = nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.GELU(),
                nn.Dropout(0.20),
            )
            self.count_heads = nn.ModuleDict({
                name: nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
                for name in self.categories
            })

        def forward(self, images):
            features = self.encoder(images)
            pooled = torch.cat((self.avg_pool(features), self.max_pool(features)), dim=1).flatten(1)
            shared = self.shared(pooled)
            return {name: head(shared).squeeze(-1) for name, head in self.count_heads.items()}
else:
    class SharedFloorplanCounter:  # pragma: no cover - exercised when torch is absent
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Training or running the floor-plan CNN requires PyTorch. Install torch first.")


def require_torch():
    if torch is None:
        raise RuntimeError(
            "The floor-plan count model requires PyTorch. Install a Python/PyTorch "
            "build supported by your platform, then install Pillow for image loading."
        )
    return torch


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def question_category(question: str) -> Optional[str]:
    """Map English/Finnish floor-plan count questions to canonical categories."""
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
