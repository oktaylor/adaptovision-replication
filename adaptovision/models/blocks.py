"""Building blocks for the AdaptoVision replication model."""

from __future__ import annotations

import torch
from torch import nn


def get_activation(name: str = "elu") -> nn.Module:
    """Return activation module."""
    name = name.lower()
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class ConvNormAct(nn.Module):
    """Convolution + BatchNorm + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    """Residual-compatible depthwise separable convolution block.

    This approximates the paper's Block-2 idea:
    pointwise conv -> depthwise conv -> pointwise conv -> dropout.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or channels

        self.block = nn.Sequential(
            ConvNormAct(channels, hidden_channels, kernel_size=1, activation=activation),
            ConvNormAct(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                groups=hidden_channels,
                activation=activation,
            ),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(p=dropout),
        )
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class EnhancedResidualUnit(nn.Module):
    """Enhanced residual unit used in the AdaptoVision replication.

    The original paper describes ERUs as deeper residual transformations.
    This implementation uses a bottlenecked transformation path to keep the
    model feasible on CIFAR-10 and cluster GPU limits.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "elu",
        bottleneck_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = max(16, int(channels * bottleneck_ratio))

        self.transform = nn.Sequential(
            ConvNormAct(channels, hidden_channels, kernel_size=1, activation=activation),
            ConvNormAct(hidden_channels, hidden_channels, kernel_size=kernel_size, activation=activation),
            DepthwiseSeparableBlock(
                hidden_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                activation=activation,
            ),
            ConvNormAct(hidden_channels, channels, kernel_size=1, activation=activation),
            nn.Dropout2d(p=dropout),
        )
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.transform(x))


class DownsampleBlock(nn.Module):
    """Downsample spatial resolution and change channel width."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2, activation=activation),
            ConvNormAct(out_channels, out_channels, kernel_size=3, activation=activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HierarchicalSkipFusion(nn.Module):
    """Fuse current features with projected earlier features.

    This is a practical implementation of hierarchical skip connections:
    earlier feature maps are pooled/interpolated and projected to the current
    channel dimension before addition.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = torch.nn.functional.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.proj(source)


class EncoderStage(nn.Module):
    """One hierarchical encoder stage."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        dropout: float,
        activation: str = "elu",
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                EnhancedResidualUnit(
                    channels=channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
