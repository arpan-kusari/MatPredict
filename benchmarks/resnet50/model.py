import torch
import torch.nn as nn
import torchvision.models as models


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


class ResNet50UNet(nn.Module):
    """U-Net style decoder with ResNet-50 encoder for object-shaped PBR prediction.

    Two independent prediction heads:
      - pbr: 5 channels (albedo RGB + roughness + metallic)
      - material_logits: num_material_classes channels (optional)
    """

    def __init__(
        self,
        out_channels: int = 4,
        pretrained: bool = True,
        num_material_classes: int = 0,
        enable_pbr_head: bool = True,
    ):
        super().__init__()
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        backbone = models.resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.bottleneck = ConvBlock(2048, 1024)
        self.up4 = UpBlock(1024, 1024, 512)
        self.up3 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up1 = UpBlock(128, 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(32, 32)
        self.pbr_head = nn.Conv2d(32, out_channels, kernel_size=1) if enable_pbr_head else None
        self.material_head = nn.Conv2d(32, num_material_classes, kernel_size=1) if num_material_classes > 0 else None

    def forward(self, x):
        x0 = self.stem(x)            # 112x112
        x1 = self.layer1(self.maxpool(x0))  # 56x56
        x2 = self.layer2(x1)         # 28x28
        x3 = self.layer3(x2)         # 14x14
        x4 = self.layer4(x3)         # 7x7

        x = self.bottleneck(x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)

        x = self.final_up(x)
        x = self.final_conv(x)

        pbr = self.pbr_head(x) if self.pbr_head is not None else None
        material_logits = self.material_head(x) if self.material_head is not None else None

        if material_logits is None and pbr is not None:
            return pbr
        if pbr is None and material_logits is not None:
            return material_logits
        return {"pbr": pbr, "material_logits": material_logits}
