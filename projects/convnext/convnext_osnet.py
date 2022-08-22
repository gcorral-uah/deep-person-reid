# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional
from .drop import DropPath


class StackedConv(nn.Module):
    """Lightweight convolution stream."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        kernel_stride: int,
        kernel_padding: int,
        groups: int,
        depth=1,
    ):
        super(StackedConv, self).__init__()
        layers = []
        self.out_channels = 0

        for _ in range(depth):
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=kernel_stride,
                padding=kernel_padding,
                groups=groups,
            )  # depthwise conv
            layers.append(conv)

            self.out_channels += conv.out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps=1e-6,
        data_format="channels_last",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)  # type: ignore
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ChannelGate(nn.Module):
    r"""A mini-network that generates channel-wise gates conditioned on input tensor.
    It allows the mix of the result of multiple networks depending on the input image.
    """

    def __init__(
        self,
        in_channels: int,
        num_gates: Optional[int] = None,
        return_gates=False,
        gate_activation="sigmoid",
        reduction=16,
        layer_norm=False,
    ):
        super(ChannelGate, self).__init__()

        if num_gates is None:
            num_gates = in_channels

        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(
            in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0
        )
        if gate_activation == "sigmoid":
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == "relu":
            self.gate_activation = nn.ReLU()
        elif gate_activation == "linear":
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x: torch.Tensor):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class MultiBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        kernel_size (int): Sizes of the convolutional kernels of the convnext block. Default: 7
        kernel_stride (int): Strides of the convolutional kernels of the convnext block. Default: 0
        kernel_padding (int): Padding of the convolutional kernels of the convnext block. Default: 3
        num_conv_blocks (int): Number of convulutioinal blocks that we pile to create a greater receptive file. Default: 1
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim: int,
        kernel_size=7,
        kernel_stride=0,
        kernel_padding=3,
        num_conv_blocks=1,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        self.conv_layers_stack = nn.ModuleList()
        stacked_conv_total_out_channels = 0
        for num_blocks in range(1, num_conv_blocks + 1):
            self.dwconv = StackedConv(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                kernel_stride=kernel_stride,
                kernel_padding=kernel_padding,
                groups=dim,
                depth=num_blocks,
            )  # depthwise conv
            self.conv_layers_stack.append(self.dwconv)
            stacked_conv_total_out_channels += self.dwconv.out_channels

        reduction_factor_out_channels = stacked_conv_total_out_channels // dim
        self.gate = ChannelGate(
            in_channels=dim, reduction=reduction_factor_out_channels
        )

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        input = x
        # x = self.dwconv(x) # Old code when we only had one convolution
        x_conv = 0
        for convolution in self.conv_layers_stack:
            x_conv_t = convolution(x)
            x_conv = x_conv + self.gate(x_conv_t)
        x = x_conv  # type: ignore

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtOSNet(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`, with added ideas from OSNet.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        kernel_sizes (int): Sizes of the convolutional kernels of the two downsample blocks (1st and 2nd element) and the convnext blocks (3th element). Default: [4, 2, 7]
        kernel_strides (int): Strides of the convolutional kernels of the two downsample blocks (1st and 2nd element) and the convnext blocks (3th element). Default: [4, 2, 1]
        kernel_paddings (int): Paddings of the convolutional kernels of the two downsample blocks (1st and 2nd element) and the convnext blocks (3th element). Default: [0,0,3]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        kernel_sizes: List[int] = [4, 2, 7],
        kernel_strides: List[int] = [4, 2, 1],
        kernel_paddings: List[int] = [0, 0, 3],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(
                in_chans,
                dims[0],
                kernel_size=kernel_sizes[0],
                stride=kernel_strides[0],
                padding=kernel_paddings[0],
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=kernel_sizes[1],
                    stride=kernel_strides[1],
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    MultiBlock(
                        dim=dims[i],
                        kernel_size=kernel_sizes[2],
                        kernel_stride=kernel_strides[2],
                        kernel_padding=kernel_paddings[2],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m: torch.nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # This if is needed becase the bias term is nn.Module is defined as Optional[Tensor], but we can't initialize a None value
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.head(x)
        return x
