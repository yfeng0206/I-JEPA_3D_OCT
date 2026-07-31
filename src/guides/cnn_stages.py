"""Frozen CNN stage extractors."""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    ResNeXt101_32X8D_Weights,
    convnext_base,
    convnext_tiny,
    resnet50,
    resnext101_32x8d,
)


class _ResNetStages(nn.Module):
    expected_shapes = {
        "stage1": (256, 56, 56),
        "stage2": (512, 28, 28),
        "stage3": (1024, 14, 14),
        "stage4": (2048, 7, 7),
    }

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.stage1 = base.layer1
        self.stage2 = base.layer2
        self.stage3 = base.layer3
        self.stage4 = base.layer4
        self.requires_grad_(False)
        self.eval()

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self.stem(x)
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        return {
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "stage4": stage4,
        }


class ResNet50Stages(_ResNetStages):
    """Frozen ResNet-50 returning only Stage 1-4 feature maps."""

    name = "resnet50"
    weights_name = "ResNet50_Weights.IMAGENET1K_V2"

    def __init__(self) -> None:
        super().__init__(
            resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V2,
                progress=False,
            )
        )


class ResNeXt10132X8DStages(_ResNetStages):
    """Frozen ResNeXt-101 32x8d returning Stage 1-4 feature maps."""

    name = "resnext101_32x8d"
    weights_name = "ResNeXt101_32X8D_Weights.IMAGENET1K_V2"

    def __init__(self) -> None:
        super().__init__(
            resnext101_32x8d(
                weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2,
                progress=False,
            )
        )


class _ConvNeXtStages(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.stem = base.features[0]
        self.stage1 = base.features[1]
        self.downsample2 = base.features[2]
        self.stage2 = base.features[3]
        self.downsample3 = base.features[4]
        self.stage3 = base.features[5]
        self.downsample4 = base.features[6]
        self.stage4 = base.features[7]
        self.requires_grad_(False)
        self.eval()

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self.stem(x)
        stage1 = self.stage1(x)
        stage2 = self.stage2(self.downsample2(stage1))
        stage3 = self.stage3(self.downsample3(stage2))
        stage4 = self.stage4(self.downsample4(stage3))
        return {
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "stage4": stage4,
        }


class ConvNeXtTinyStages(_ConvNeXtStages):
    """Frozen ConvNeXt-Tiny returning only Stage 1-4 feature maps."""

    name = "convnext_tiny"
    weights_name = "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
    expected_shapes = {
        "stage1": (96, 56, 56),
        "stage2": (192, 28, 28),
        "stage3": (384, 14, 14),
        "stage4": (768, 7, 7),
    }

    def __init__(self) -> None:
        super().__init__(
            convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                progress=False,
            )
        )


class ConvNeXtBaseStages(_ConvNeXtStages):
    """Frozen ConvNeXt-Base returning only Stage 1-4 feature maps."""

    name = "convnext_base"
    weights_name = "ConvNeXt_Base_Weights.IMAGENET1K_V1"
    expected_shapes = {
        "stage1": (128, 56, 56),
        "stage2": (256, 28, 28),
        "stage3": (512, 14, 14),
        "stage4": (1024, 7, 7),
    }

    def __init__(self) -> None:
        super().__init__(
            convnext_base(
                weights=ConvNeXt_Base_Weights.IMAGENET1K_V1,
                progress=False,
            )
        )
