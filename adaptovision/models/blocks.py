"""Building blocks for the AdaptoVision replication model.

This implementation follows the architectural components explicitly described
in the AdaptoVision paper:

1. Enhanced Residual Unit:
   y = x + T(x), where T contains four convolution kernels and Block-2.

2. Block-2:
   pointwise convolution -> depthwise convolution -> pointwise convolution.

3. Encoder transition:
   downsampling followed by global average pooling, reshape, and 1x1 projection.

4. Hierarchical skip fusion:
   each stage can receive projected features from the previous two stage outputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


ActivationName = str


def get_activation(name: ActivationName = "elu") -> nn.Module:
    """Return activation module."""
    name = name.lower()

    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)

    raise ValueError(f"Unsupported activation: {name}")


class ConvNormAct(nn.Module):
    """Convolution + BatchNorm + optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: str = "elu",
        use_activation: bool = True,
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
            get_activation(activation) if use_activation else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Block2(nn.Module):
    """AdaptoVision Block-2.

    Paper form:
        B2(z) = sigma(Wb * D(sigma(Wa * z)))

    where:
        Wa, Wb: 1x1 pointwise convolutions
        D: depthwise convolution

    The paper figure also presents Block-2 as residual-compatible, so this
    implementation keeps an internal residual addition when shape is unchanged.
    """

    def __init__(
        self,
        channels: int,
        depthwise_kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        self.pointwise_a = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            activation=activation,
            use_activation=True,
        )

        self.depthwise = ConvNormAct(
            channels,
            channels,
            kernel_size=depthwise_kernel_size,
            groups=channels,
            activation=activation,
            use_activation=True,
        )

        self.pointwise_b = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            activation=activation,
            use_activation=False,
        )

        self.dropout = nn.Dropout2d(p=dropout)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pointwise_a(x)
        out = self.depthwise(out)
        out = self.pointwise_b(out)
        out = self.dropout(out)

        return self.activation(x + out)


class EnhancedResidualUnit(nn.Module):
    """Enhanced Residual Unit following the AdaptoVision paper.

    Paper form:
        y_l = x_{l-1} + T(x_{l-1})

        T(x) = sigma W4 * (
                   sigma W3 * B2(
                       sigma W2 * (
                           sigma W1 * x
                       )
                   )
               )

    This implementation uses:
        W1: 1x1 conv
        W2: kxk conv
        B2: pointwise-depthwise-pointwise block
        W3: kxk conv
        W4: 1x1 conv
    """

    def __init__(
        self,
        channels: int,
        depthwise_kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        self.w1 = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            activation=activation,
            use_activation=True,
        )

        self.w2 = ConvNormAct(
            channels,
            channels,
            kernel_size=depthwise_kernel_size,
            activation=activation,
            use_activation=True,
        )

        self.block2 = Block2(
            channels=channels,
            depthwise_kernel_size=depthwise_kernel_size,
            dropout=dropout,
            activation=activation,
        )

        self.w3 = ConvNormAct(
            channels,
            channels,
            kernel_size=depthwise_kernel_size,
            activation=activation,
            use_activation=True,
        )

        self.w4 = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            activation=activation,
            use_activation=False,
        )

        self.dropout = nn.Dropout2d(p=dropout)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transform = self.w1(x)
        transform = self.w2(transform)
        transform = self.block2(transform)
        transform = self.w3(transform)
        transform = self.w4(transform)
        transform = self.dropout(transform)

        return self.activation(x + transform)


class EncoderStage(nn.Module):
    """Stack of Enhanced Residual Units."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        depthwise_kernel_size: int,
        dropout: float,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                EnhancedResidualUnit(
                    channels=channels,
                    depthwise_kernel_size=depthwise_kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class EncoderTransition(nn.Module):
    """Stage transition with downsampling and global context projection.

    Paper form:
        x^{k+1} = S_k(Reshape(GAP(P_k(x^k))))

    To preserve spatial feature maps for later convolutional stages, this module
    combines:
        local downsampled feature map
        +
        broadcasted global context from GAP -> 1x1 projection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.local_projection = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=3,
            activation=activation,
            use_activation=True,
        )

        self.global_projection = ConvNormAct(
            out_channels,
            out_channels,
            kernel_size=1,
            activation=activation,
            use_activation=False,
        )

        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.downsample(x)
        local = self.local_projection(local)

        global_context = F.adaptive_avg_pool2d(local, output_size=(1, 1))
        global_context = self.global_projection(global_context)
        global_context = global_context.expand_as(local)

        return self.activation(local + global_context)


class HierarchicalSkipFusion(nn.Module):
    """Hierarchical skip fusion from the previous two stage outputs.

    Paper form:
        x_out^(k) = alpha1 * x^(k-1) + alpha2 * x^(k-2) + F(x^(k-1))

    Since feature maps have different channel counts and resolutions across
    stages, previous outputs are projected with 1x1 convolutions and resized to
    the current spatial size before addition.
    """

    def __init__(
        self,
        prev1_channels: int,
        out_channels: int,
        prev2_channels: int | None = None,
        activation: str = "elu",
        learnable_weights: bool = True,
    ) -> None:
        super().__init__()

        self.prev1_projection = ConvNormAct(
            prev1_channels,
            out_channels,
            kernel_size=1,
            activation=activation,
            use_activation=False,
        )

        self.prev2_projection = None
        if prev2_channels is not None:
            self.prev2_projection = ConvNormAct(
                prev2_channels,
                out_channels,
                kernel_size=1,
                activation=activation,
                use_activation=False,
            )

        if learnable_weights:
            self.alpha1 = nn.Parameter(torch.tensor(1.0))
            self.alpha2 = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("alpha1", torch.tensor(1.0))
            self.register_buffer("alpha2", torch.tensor(1.0))

    def _project_and_resize(
        self,
        x: torch.Tensor,
        projection: nn.Module,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        x = projection(x)

        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        return x

    def forward(
        self,
        prev1: torch.Tensor,
        prev2: torch.Tensor | None,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        fused = self.alpha1 * self._project_and_resize(
            prev1,
            self.prev1_projection,
            target_size,
        )

        if prev2 is not None and self.prev2_projection is not None:
            fused = fused + self.alpha2 * self._project_and_resize(
                prev2,
                self.prev2_projection,
                target_size,
            )

        return fused