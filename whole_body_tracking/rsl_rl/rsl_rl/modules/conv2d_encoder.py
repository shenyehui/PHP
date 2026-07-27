# Copyright (c) 2025, Whole Body Tracking Contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Conv2D encoder for depth image processing in student policies."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class Conv2dEncoder(nn.Module):
    """A lightweight Conv2D encoder that processes depth images into a flat latent vector.

    Architecture:
        depth_image (C, H, W) → Conv2D layers → (GAP or Flatten) → MLP → latent vector
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 18,
        input_width: int = 32,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        strides: list[int] | None = None,
        paddings: list[int] | None = None,
        hidden_sizes: list[int] | None = None,
        output_size: int = 32,
        activation: str = "relu",
        use_maxpool: bool = False,
        global_average_pool: bool = False,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 32]
        if kernel_sizes is None:
            kernel_sizes = [3, 3]
        if strides is None:
            strides = [1, 1]
        if paddings is None:
            paddings = [1, 1]
        if hidden_sizes is None:
            hidden_sizes = [32]

        activation_fn = resolve_nn_activation(activation)

        # Build Conv2D layers
        conv_layers = []
        in_c = input_channels
        for i, (out_c, k, s, p) in enumerate(zip(channels, kernel_sizes, strides, paddings)):
            conv_layers.append(nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p))
            conv_layers.append(activation_fn)
            if use_maxpool:
                conv_layers.append(nn.MaxPool2d(kernel_size=2))
            in_c = out_c
        self.conv = nn.Sequential(*conv_layers)

        if not (len(channels) == len(kernel_sizes) == len(strides) == len(paddings)):
            raise ValueError("channels, kernel_sizes, strides and paddings must have the same length")

        # Compute flattened size after conv layers
        h, w = input_height, input_width
        for k, s, p in zip(kernel_sizes, strides, paddings):
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1
            if use_maxpool:
                h //= 2
                w //= 2
        self.global_average_pool = global_average_pool
        self.flattened_size = in_c if global_average_pool else in_c * h * w

        # Build MLP head
        mlp_layers = []
        mlp_in = self.flattened_size
        for hs in hidden_sizes:
            mlp_layers.append(nn.Linear(mlp_in, hs))
            mlp_layers.append(activation_fn)
            mlp_in = hs
        mlp_layers.append(nn.Linear(mlp_in, output_size))
        self.mlp = nn.Sequential(*mlp_layers)

        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode depth image into latent vector.

        Args:
            x: Depth image tensor of shape (B, C, H, W)

        Returns:
            Latent vector of shape (B, output_size)
        """
        x = self.conv(x)
        if self.global_average_pool:
            x = x.mean(dim=(-2, -1))
        else:
            x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x
