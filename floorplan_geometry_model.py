"""Multi-task floor-plan perception model: room masks, icon masks, and counts."""

from __future__ import annotations

from typing import Iterable

from floorplan_model import CATEGORY_NAMES


ROOM_CLASSES = (
    "background", "outdoor", "wall", "kitchen", "living_room", "bedroom",
    "bathroom", "entry", "railing", "storage", "garage", "undefined",
)
ICON_CLASSES = (
    "no_icon", "window", "door", "closet", "electrical_appliance", "toilet",
    "sink", "sauna_bench", "fireplace", "bathtub", "chimney",
)
INSTANCE_ICON_CLASSES = ICON_CLASSES[1:]
GEOMETRY_MODEL_VERSION = 1

# The CNN's pooled count heads are auxiliary. At inference, counts for mapped
# categories come from connected components in the predicted segmentation.
ICON_COUNT_CLASS = {
    "windows": "window",
    "doors": "door",
    "toilet_bowls": "toilet",
    "sinks": "sink",
    "showers_bathtubs": "bathtub",
}
ROOM_COUNT_CLASS = {
    "bedrooms": "bedroom",
    "kitchens": "kitchen",
    "bathrooms": "bathroom",
}


try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


if nn is not None:
    class _ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, out_channels),
            )
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.activation = nn.GELU()

        def forward(self, x):
            return self.activation(self.main(x) + self.skip(x))


    class _UpBlock(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
            super().__init__()
            self.block = _ConvBlock(in_channels + skip_channels, out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.block(torch.cat((x, skip), dim=1))


    class FloorplanGeometryNet(nn.Module):
        """Shared CNN with segmentation decoders and an auxiliary count branch."""

        def __init__(self, count_categories: Iterable[str] = CATEGORY_NAMES):
            super().__init__()
            self.count_categories = tuple(count_categories)
            self.enc1 = _ConvBlock(3, 32)
            self.enc2 = _ConvBlock(32, 64)
            self.enc3 = _ConvBlock(64, 128)
            self.enc4 = _ConvBlock(128, 192)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _ConvBlock(192, 256)

            self.up4 = _UpBlock(256, 192, 192)
            self.up3 = _UpBlock(192, 128, 128)
            self.up2 = _UpBlock(128, 64, 64)
            self.up1 = _UpBlock(64, 32, 32)
            self.room_head = nn.Conv2d(32, len(ROOM_CLASSES), kernel_size=1)
            self.icon_head = nn.Conv2d(32, len(ICON_CLASSES), kernel_size=1)

            self.count_features = nn.Sequential(
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2)
            )
            self.count_heads = nn.ModuleDict({
                name: nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
                for name in self.count_categories
            })

        def forward(self, images):
            e1 = self.enc1(images)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            center = self.bottleneck(self.pool(e4))

            d4 = self.up4(center, e4)
            d3 = self.up3(d4, e3)
            d2 = self.up2(d3, e2)
            d1 = self.up1(d2, e1)
            room_logits = self.room_head(d1)
            icon_logits = self.icon_head(d1)

            pooled = torch.cat((F.adaptive_avg_pool2d(center, 1), F.adaptive_max_pool2d(center, 1)), dim=1).flatten(1)
            count_features = self.count_features(pooled)
            count_log1p = torch.cat(
                [self.count_heads[name](count_features) for name in self.count_categories], dim=1
            )
            return {"rooms": room_logits, "icons": icon_logits, "count_log1p": count_log1p}
else:
    class FloorplanGeometryNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("The geometry model requires PyTorch.")
