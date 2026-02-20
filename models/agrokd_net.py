"""
AgroKD-Net: Energy-Aware Lightweight Crop–Weed Detection Network.

Architecture overview
---------------------
* **AgroKDBackbone** — 5-stage depthwise-separable backbone with
  Squeeze-and-Excitation (SE) attention in the deeper stages.
* **LightFPN** — lightweight top-down Feature Pyramid Network.
* **AgroKDNet** — backbone + FPN + 3 detection heads (small / medium / large).

The design targets low-energy inference on edge devices while remaining
compatible with knowledge distillation from heavier teacher models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution block.

    Depthwise conv → Pointwise conv → BatchNorm → SiLU.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    stride : int
        Applied to the depthwise conv.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel-attention block.

    Parameters
    ----------
    channels : int
        Number of input (and output) channels.
    reduction : int
        Channel reduction ratio for the bottleneck FC layers (default: 16).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class AgroKDBackbone(nn.Module):
    """
    5-stage depthwise-separable backbone.

    Returns three feature maps at strides 8, 16, and 32 (f3, f4, f5) for
    use by the FPN head.

    Channel progression: 3 → 32 → 64 → 128 → 256 → 512
    """

    def __init__(self) -> None:
        super().__init__()

        # Stage 1: standard 3×3 conv (stride 2) — 3 → 32
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # Stage 2: 32 → 64 (stride 2)
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64),
        )

        # Stage 3: 64 → 128 (stride 2) + SE
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            SEBlock(128),
        )

        # Stage 4: 128 → 256 (stride 2) + SE
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
            SEBlock(256),
        )

        # Stage 5: 256 → 512 (stride 2) + SE
        self.stage5 = nn.Sequential(
            DepthwiseSeparableConv(256, 512, stride=2),
            DepthwiseSeparableConv(512, 512),
            SEBlock(512),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        f3, f4, f5 : torch.Tensor
            Feature maps at stride-8, stride-16, and stride-32 respectively.
        """
        s1 = self.stage1(x)   # stride 2
        s2 = self.stage2(s1)  # stride 4
        f3 = self.stage3(s2)  # stride 8
        f4 = self.stage4(f3)  # stride 16
        f5 = self.stage5(f4)  # stride 32
        return f3, f4, f5


# ---------------------------------------------------------------------------
# Lightweight FPN
# ---------------------------------------------------------------------------


class LightFPN(nn.Module):
    """
    Lightweight top-down Feature Pyramid Network.

    * Lateral 1×1 convolutions project each backbone level to *out_channels*.
    * Top-down pathway: upsample + add.
    * Smooth layers refine each merged level with a DepthwiseSeparableConv.
    """

    def __init__(self, in_channels: Tuple[int, int, int] = (128, 256, 512), out_channels: int = 128) -> None:
        super().__init__()
        c3, c4, c5 = in_channels

        # Lateral projections
        self.lat5 = nn.Conv2d(c5, out_channels, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(c4, out_channels, kernel_size=1, bias=False)
        self.lat3 = nn.Conv2d(c3, out_channels, kernel_size=1, bias=False)

        # Smooth layers
        self.smooth5 = DepthwiseSeparableConv(out_channels, out_channels)
        self.smooth4 = DepthwiseSeparableConv(out_channels, out_channels)
        self.smooth3 = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(
        self,
        f3: torch.Tensor,
        f4: torch.Tensor,
        f5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        p3, p4, p5 : torch.Tensor
            FPN outputs at stride 8, 16, 32 with uniform channel width.
        """
        p5 = self.smooth5(self.lat5(f5))
        p4 = self.smooth4(
            self.lat4(f4) + F.interpolate(p5, size=f4.shape[-2:], mode="nearest")
        )
        p3 = self.smooth3(
            self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="nearest")
        )
        return p3, p4, p5


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class AgroKDNet(nn.Module):
    """
    AgroKD-Net: full lightweight detection model.

    Parameters
    ----------
    num_classes : int
        Number of target object classes (default: 15 for CottonWeed).
    fpn_channels : int
        Uniform channel width inside the FPN (default: 128).
    """

    def __init__(self, num_classes: int = 15, fpn_channels: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.backbone = AgroKDBackbone()
        self.fpn = LightFPN(in_channels=(128, 256, 512), out_channels=fpn_channels)

        # Detection heads: one per FPN level
        # Output channels = 5 + num_classes  (tx, ty, tw, th, objectness, classes)
        out_ch = 5 + num_classes
        self.head_small = self._make_head(fpn_channels, out_ch)   # p3 (small objs)
        self.head_medium = self._make_head(fpn_channels, out_ch)  # p4
        self.head_large = self._make_head(fpn_channels, out_ch)   # p5 (large objs)

    @staticmethod
    def _make_head(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            DepthwiseSeparableConv(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 3, H, W)``.

        Returns
        -------
        out_small, out_medium, out_large : torch.Tensor
            Detection maps at strides 8, 16, 32 with shape
            ``(B, 5+num_classes, H/stride, W/stride)``.
        """
        f3, f4, f5 = self.backbone(x)
        p3, p4, p5 = self.fpn(f3, f4, f5)

        out_small = self.head_small(p3)
        out_medium = self.head_medium(p4)
        out_large = self.head_large(p5)

        return out_small, out_medium, out_large


# ---------------------------------------------------------------------------
# Self-test / parameter report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")

    model = AgroKDNet(num_classes=15).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters    : {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")

    # Forward pass
    dummy = torch.randn(2, 3, 640, 640, device=device)
    out_s, out_m, out_l = model(dummy)
    print(f"✅ Output shapes (B=2, img=640):")
    print(f"   small  (stride-8)  : {tuple(out_s.shape)}")
    print(f"   medium (stride-16) : {tuple(out_m.shape)}")
    print(f"   large  (stride-32) : {tuple(out_l.shape)}")

    # FLOPs (requires thop)
    try:
        from thop import profile  # type: ignore

        single = torch.randn(1, 3, 640, 640, device=device)
        flops, _ = profile(model, inputs=(single,), verbose=False)
        print(f"📊 FLOPs (640×640): {flops / 1e9:.2f} G")
    except ImportError:
        print("ℹ️  Install thop (`pip install thop`) to compute FLOPs.")
    except Exception as exc:
        print(f"⚠️  FLOPs computation failed: {exc}")

    print("✅ AgroKDNet self-test complete.")
    sys.exit(0)
