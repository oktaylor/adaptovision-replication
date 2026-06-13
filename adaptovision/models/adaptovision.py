"""AdaptoVision model implementation for CIFAR-style image classification."""

from __future__ import annotations

import torch
from torch import nn

from adaptovision.models.blocks import (
    ConvNormAct,
    DownsampleBlock,
    EncoderStage,
    HierarchicalSkipFusion,
)


class AdaptoVision(nn.Module):
    """CNN architecture inspired by the AdaptoVision paper.

    This implementation focuses on the main architectural ideas:
    enhanced residual units, depthwise separable blocks, and hierarchical
    skip connections. It is designed for from-scratch CIFAR-10 replication.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        stage_channels: list[int] | tuple[int, ...] = (64, 128, 256, 512),
        blocks_per_stage: list[int] | tuple[int, ...] = (2, 2, 2, 2),
        dropout_rates: list[float] | tuple[float, ...] = (0.30, 0.35, 0.40, 0.50),
        activation: str = "elu",
    ) -> None:
        super().__init__()

        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError("stage_channels and blocks_per_stage must have the same length.")
        if len(stage_channels) != len(dropout_rates):
            raise ValueError("stage_channels and dropout_rates must have the same length.")

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, base_channels, kernel_size=3, activation=activation),
            ConvNormAct(base_channels, stage_channels[0], kernel_size=3, activation=activation),
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()

        for i, channels in enumerate(stage_channels):
            self.stages.append(
                EncoderStage(
                    channels=channels,
                    num_blocks=blocks_per_stage[i],
                    dropout=dropout_rates[i],
                    activation=activation,
                    kernel_size=3 if i < 2 else 5,
                )
            )

            if i < len(stage_channels) - 1:
                self.downsamples.append(
                    DownsampleBlock(
                        in_channels=stage_channels[i],
                        out_channels=stage_channels[i + 1],
                        activation=activation,
                    )
                )

            if i >= 1:
                self.skip_fusions.append(
                    HierarchicalSkipFusion(
                        in_channels=stage_channels[i - 1],
                        out_channels=stage_channels[i],
                    )
                )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(stage_channels[-1], num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        previous = None
        for i, stage in enumerate(self.stages):
            x = stage(x)

            if previous is not None:
                fusion = self.skip_fusions[i - 1](previous, x)
                x = x + fusion

            previous = x

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
