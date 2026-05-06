import timm
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _to_nchw(feat: torch.Tensor, expected_channels: int) -> torch.Tensor:
    if feat.ndim != 4:
        raise ValueError(f"Expected 4D feature tensor, got shape={tuple(feat.shape)}")

    # Accept both NCHW and NHWC robustly using encoder-reported channels.
    if feat.shape[1] == expected_channels:
        return feat
    if feat.shape[-1] == expected_channels:
        return feat.permute(0, 3, 1, 2).contiguous()

    raise ValueError(
        "Cannot infer Swin feature layout. "
        f"shape={tuple(feat.shape)}, expected_channels={expected_channels}"
    )


class SwinTUNet(nn.Module):
    """U-Net style decoder with Swin-T encoder for object-shaped PBR prediction.

    Two independent prediction heads:
      - pbr: 5 channels (albedo RGB + roughness + metallic)
      - material_logits: num_material_classes channels (optional)
    """

    def __init__(
        self,
        out_channels: int = 4,
        pretrained: bool = True,
        image_size: int = 224,
        num_material_classes: int = 0,
        enable_pbr_head: bool = True,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=image_size,
        )

        c1, c2, c3, c4 = self.encoder.feature_info.channels()
        self._enc_channels = (c1, c2, c3, c4)

        self.bottleneck = ConvBlock(c4, c4)
        self.up4 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up2 = UpBlock(c2, c1, c1)

        self.up1 = nn.ConvTranspose2d(c1, 64, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(32, 32)
        self.pbr_head = nn.Conv2d(32, out_channels, kernel_size=1) if enable_pbr_head else None
        self.material_head = nn.Conv2d(32, num_material_classes, kernel_size=1) if num_material_classes > 0 else None

    def forward(self, x):
        feats = self.encoder(x)
        c1, c2, c3, c4 = self._enc_channels
        f1, f2, f3, f4 = [
            _to_nchw(feats[0], c1),
            _to_nchw(feats[1], c2),
            _to_nchw(feats[2], c3),
            _to_nchw(feats[3], c4),
        ]

        x = self.bottleneck(f4)
        x = self.up4(x, f3)
        x = self.up3(x, f2)
        x = self.up2(x, f1)

        x = self.up1(x)
        x = self.conv1(x)
        x = self.final_up(x)
        x = self.final_conv(x)
        pbr = self.pbr_head(x) if self.pbr_head is not None else None
        material_logits = self.material_head(x) if self.material_head is not None else None

        if material_logits is None and pbr is not None:
            return pbr
        if pbr is None and material_logits is not None:
            return material_logits
        return {"pbr": pbr, "material_logits": material_logits}
