"""Shared CNN with task-specific count heads for residential floor plans.

The QA pairs supervise image-level counts. They do not identify individual
objects, so this model predicts counts but does not localize them.
"""

from __future__ import annotations

from typing import Dict, Iterable


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
