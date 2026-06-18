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

class ConvBN(nn.Module):
    """Convolution + BatchNorm without activation.

    This helper is used to approximate the CN / CN_BN units in Figure 1-D.
    Activations are applied explicitly at the places where Figure 1-D shows
    an Activation block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvConvBN(nn.Module):
    """Approximation of the CN-CN_BN units shown in Figure 1-D.

    Flow:
        Conv-BN -> activation -> Conv-BN

    The second conv does not include activation because the residual ADD or
    explicit Activation block comes after it in Figure 1-D.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        self.conv1 = ConvBN(
            channels,
            channels,
            kernel_size=kernel_size,
        )
        self.activation = get_activation(activation)
        self.conv2 = ConvBN(
            channels,
            channels,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x


class Block1(nn.Module):
    """Lightweight Figure 1-D style Block-1.

    This block keeps the main Figure 1-D idea:
        - multiple internal additions
        - Block2 in the middle
        - final long skip from the original input

    But it follows the paper's ERU formula more closely:
        W1 -> W2 -> Block2 -> W3 -> W4

    Therefore it avoids the parameter explosion caused by using five
    full ConvConvBN modules.
    """

    def __init__(
        self,
        channels: int,
        depthwise_kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        k = depthwise_kernel_size

        # W1: pointwise projection
        self.w1 = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            activation=activation,
            use_activation=False,
        )

        # W2: spatial convolution
        self.w2 = ConvNormAct(
            channels,
            channels,
            kernel_size=k,
            activation=activation,
            use_activation=False,
        )

        # Block2: pointwise -> depthwise -> pointwise
        self.block2 = Block2(
            channels=channels,
            depthwise_kernel_size=k,
            dropout=dropout,
            activation=activation,
        )

        # W3: spatial convolution after Block2
        self.w3 = ConvNormAct(
            channels,
            channels,
            kernel_size=k,
            activation=activation,
            use_activation=False,
        )

        # W4: final projection
        self.w4 = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            activation=activation,
            use_activation=False,
        )

        self.mid_activation = get_activation(activation)
        self.final_activation = get_activation(activation)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # First local skip:
        # x1 = x + W2(W1(x))
        h = self.w1(x)
        h = self.w2(h)
        x1 = x + h

        # Middle skip with Block2:
        # x2 = x1 + W3(Block2(x1))
        h = self.block2(x1)
        h = self.w3(h)
        x2 = self.mid_activation(x1 + h)

        # Later local transform using W4:
        # x3 = x2 + W4(x2)
        h = self.w4(x2)
        x3 = x2 + h

        # Final activation/dropout and long skip from original input.
        h = self.final_activation(x3)
        h = self.dropout(h)

        out = identity + h
        return out


class Block2(nn.Module):
    """AdaptoVision Block-2, Eq. (5)-based.

    Strict interpretation:
        sigma = BatchNorm
        Wa, Wb = 1x1 pointwise conv
        D = depthwise conv

    Flow:
        PW-Conv -> BN -> DW-Conv -> BN -> PW-Conv -> BN -> Dropout

    No internal residual addition.
    """

    def __init__(
        self,
        channels: int,
        depthwise_kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "elu",  # kept for API compatibility, not used inside Block2
    ) -> None:
        super().__init__()

        self.pointwise_a = ConvBN(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            groups=1,
        )

        self.depthwise = ConvBN(
            in_channels=channels,
            out_channels=channels,
            kernel_size=depthwise_kernel_size,
            groups=channels,
        )

        self.pointwise_b = ConvBN(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            groups=1,
        )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pointwise_a(x)
        out = self.depthwise(out)
        out = self.pointwise_b(out)
        out = self.dropout(out)
        return out


class EncoderStage(nn.Module):
    """Stack of lightweight Figure 1-D style Block-1 modules."""

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
                Block1(
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
    """Stage transition with local downsampling and global context projection.

    Paper form:
        X_{k+1} = A(P(X_k)) + S(R(X_k))

    where:
        P: spatial downsampling
        A: local convolutional projection
        R: global average pooling + reshape
        S: 1x1 convolution for channel alignment
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
            in_channels,
            out_channels,
            kernel_size=1,
            activation=activation,
            use_activation=False,
        )

        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.downsample(x)
        local = self.local_projection(pooled)

        global_context = F.adaptive_avg_pool2d(x, output_size=(1, 1))
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