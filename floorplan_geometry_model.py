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


    class _FurukawaResidual(nn.Module):
        """Residual bottleneck block matching the published CubiCasa5K state_dict layout."""

        def __init__(self, num_in: int, num_out: int):
            super().__init__()
            self.numIn = num_in
            self.numOut = num_out
            mid = num_out // 2
            self.bn = nn.BatchNorm2d(num_in)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(num_in, mid, bias=True, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(mid)
            self.conv2 = nn.Conv2d(mid, mid, bias=True, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(mid)
            self.conv3 = nn.Conv2d(mid, num_out, bias=True, kernel_size=1)
            if num_in != num_out:
                self.conv4 = nn.Conv2d(num_in, num_out, bias=True, kernel_size=1)

        def forward(self, x):
            residual = self.conv4(x) if self.numIn != self.numOut else x
            out = self.conv1(self.relu(self.bn(x)))
            out = self.conv2(self.relu(self.bn1(out)))
            out = self.conv3(self.relu(self.bn2(out)))
            return out + residual


    class FurukawaHourglass(nn.Module):
        """44-channel hourglass floor-plan network compatible with CubiCasa5K checkpoints."""

        def __init__(self, n_classes: int = 44):
            super().__init__()
            self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU(inplace=True)
            self.r01 = _FurukawaResidual(64, 128)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.r02 = _FurukawaResidual(128, 128)
            self.r03 = _FurukawaResidual(128, 128)
            self.r04 = _FurukawaResidual(128, 256)

            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.r11_a = _FurukawaResidual(256, 256)
            self.r12_a = _FurukawaResidual(256, 256)
            self.r13_a = _FurukawaResidual(256, 256)

            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.r21_a = _FurukawaResidual(256, 256)
            self.r22_a = _FurukawaResidual(256, 256)
            self.r23_a = _FurukawaResidual(256, 256)

            self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.r31_a = _FurukawaResidual(256, 256)
            self.r32_a = _FurukawaResidual(256, 256)
            self.r33_a = _FurukawaResidual(256, 256)

            self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.r41_a = _FurukawaResidual(256, 256)
            self.r42_a = _FurukawaResidual(256, 256)
            self.r43_a = _FurukawaResidual(256, 256)
            self.r44_a = _FurukawaResidual(256, 512)
            self.r45_a = _FurukawaResidual(512, 512)
            self.upsample4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

            self.r41_b = _FurukawaResidual(256, 256)
            self.r42_b = _FurukawaResidual(256, 256)
            self.r43_b = _FurukawaResidual(256, 512)

            self.r4_ = _FurukawaResidual(512, 512)
            self.upsample3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

            self.r31_b = _FurukawaResidual(256, 256)
            self.r32_b = _FurukawaResidual(256, 256)
            self.r33_b = _FurukawaResidual(256, 512)

            self.r3_ = _FurukawaResidual(512, 512)
            self.upsample2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

            self.r21_b = _FurukawaResidual(256, 256)
            self.r22_b = _FurukawaResidual(256, 256)
            self.r23_b = _FurukawaResidual(256, 512)

            self.r2_ = _FurukawaResidual(512, 512)
            self.upsample1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

            self.r11_b = _FurukawaResidual(256, 256)
            self.r12_b = _FurukawaResidual(256, 256)
            self.r13_b = _FurukawaResidual(256, 512)

            self.conv2_ = nn.Conv2d(512, 512, bias=True, kernel_size=1)
            self.bn2 = nn.BatchNorm2d(512)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3_ = nn.Conv2d(512, 256, bias=True, kernel_size=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.relu3 = nn.ReLU(inplace=True)
            self.conv4_ = nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
            self.upsample = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
            self.sigmoid = nn.Sigmoid()

        @staticmethod
        def _upsample_add(x, y):
            if y.shape != x.shape:
                return F.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=False) + y
            return x + y

        def forward(self, x):
            out = self.maxpool(self.relu1(self.bn1(self.conv1_(x))))
            out = self.r04(self.r03(self.r02(self.r01(out))))

            out1a = self.r13_a(self.r12_a(self.r11_a(self.maxpool1(out))))
            out1b = self.r13_b(self.r12_b(self.r11_b(out)))

            out2a = self.r23_a(self.r22_a(self.r21_a(self.maxpool2(out1a))))
            out2b = self.r23_b(self.r22_b(self.r21_b(out1a)))

            out3a = self.r33_a(self.r32_a(self.r31_a(self.maxpool3(out2a))))
            out3b = self.r33_b(self.r32_b(self.r31_b(out2a)))

            out4a = self.r45_a(self.r44_a(self.r43_a(self.r42_a(self.r41_a(self.maxpool4(out3a))))))
            out4b = self.r43_b(self.r42_b(self.r41_b(out3a)))

            out4 = self.r4_(self._upsample_add(self.upsample4(out4a), out4b))
            out3 = self.r3_(self._upsample_add(self.upsample3(out4), out3b))
            out2 = self.r2_(self._upsample_add(self.upsample2(out3), out2b))
            out = self._upsample_add(self.upsample1(out2), out1b)

            out = self.relu2(self.bn2(self.conv2_(out)))
            out = self.relu3(self.bn3(self.conv3_(out)))
            out = self.upsample(self.conv4_(out))
            out[:, :21] = self.sigmoid(out[:, :21])
            return out
else:
    class FloorplanGeometryNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("The geometry model requires PyTorch.")

    class FurukawaHourglass:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("The geometry model requires PyTorch.")
