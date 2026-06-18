"""AdaptoVision model implementation for CIFAR-style image classification."""

from __future__ import annotations

import torch
from torch import nn

from adaptovision.models.blocks import (
    ConvNormAct,
    EncoderStage,
    EncoderTransition,
    HierarchicalSkipFusion,
)


class AdaptoVision(nn.Module):
    """AdaptoVision model implementation for CIFAR-style image classification.

    Main components:
    - Lightweight Block-1 / ERU-inspired stage blocks
    - Eq. (5)-based Block-2 with pointwise-depthwise-pointwise convolution
    - Hierarchical skip fusion from the previous two stage outputs
    - Stage transition using downsampling + GAP + 1x1 projection
    - Fixed channel schedule selected to match the reported ~6.6M parameter scale
    - Progressive dropout and ELU activation outside the strict sigma=BN paths
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 20,
        stage_channels: list[int] | tuple[int, ...] = (20, 40, 80, 154),
        blocks_per_stage: list[int] | tuple[int, ...] = (2, 2, 2, 2),
        dropout_rates: list[float] | tuple[float, ...] = (0.30, 0.35, 0.40, 0.50),
        activation: str = "elu",
        depthwise_kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7, 7),
        learnable_skip_weights: bool = True,
    ) -> None:
        super().__init__()

        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError("stage_channels and blocks_per_stage must have the same length.")
        if len(stage_channels) != len(dropout_rates):
            raise ValueError("stage_channels and dropout_rates must have the same length.")
        if len(stage_channels) != len(depthwise_kernel_sizes):
            raise ValueError("stage_channels and depthwise_kernel_sizes must have the same length.")

        self.stem = nn.Sequential(
            ConvNormAct(
                in_channels,
                base_channels,
                kernel_size=3,
                activation=activation,
                use_activation=True,
            ),
            ConvNormAct(
                base_channels,
                stage_channels[0],
                kernel_size=3,
                activation=activation,
                use_activation=True,
            ),
        )

        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()

        for stage_idx, channels in enumerate(stage_channels):
            self.stages.append(
                EncoderStage(
                    channels=channels,
                    num_blocks=blocks_per_stage[stage_idx],
                    depthwise_kernel_size=depthwise_kernel_sizes[stage_idx],
                    dropout=dropout_rates[stage_idx],
                    activation=activation,
                )
            )

            if stage_idx < len(stage_channels) - 1:
                self.transitions.append(
                    EncoderTransition(
                        in_channels=stage_channels[stage_idx],
                        out_channels=stage_channels[stage_idx + 1],
                        activation=activation,
                    )
                )

            if stage_idx >= 1:
                prev1_channels = stage_channels[stage_idx - 1]
                prev2_channels = stage_channels[stage_idx - 2] if stage_idx >= 2 else None

                self.skip_fusions.append(
                    HierarchicalSkipFusion(
                        prev1_channels=prev1_channels,
                        prev2_channels=prev2_channels,
                        out_channels=channels,
                        activation=activation,
                        learnable_weights=learnable_skip_weights,
                    )
                )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Paper-style final projection after global aggregation.
        self.classifier = nn.Conv2d(
            stage_channels[-1],
            num_classes,
            kernel_size=1,
            bias=True,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        stage_outputs: list[torch.Tensor] = []

        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)

            # Hierarchical skip fusion from previous two stage outputs.
            if stage_idx >= 1:
                prev1 = stage_outputs[-1]
                prev2 = stage_outputs[-2] if len(stage_outputs) >= 2 else None

                skip = self.skip_fusions[stage_idx - 1](
                    prev1=prev1,
                    prev2=prev2,
                    target_size=x.shape[-2:],
                )

                x = x + skip

            stage_outputs.append(x)

            if stage_idx < len(self.transitions):
                x = self.transitions[stage_idx](x)

        x = self.global_pool(x)
        x = self.classifier(x)
        x = torch.flatten(x, start_dim=1)

        return x


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)