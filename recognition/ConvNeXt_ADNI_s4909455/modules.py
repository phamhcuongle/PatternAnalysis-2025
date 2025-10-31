"""Minimal ConvNeXt-S implementation tailored for binary classification.

Includes:
- Channel-order-aware LayerNorm for 2D tensors
- Depthwise conv blocks with layer scale and stochastic depth
- Optional legacy two-stage classification head (adapter) support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    """LayerNorm supporting both channels_last (NHWC) and channels_first (NCHW)."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    """ConvNeXt block: DWConv -> (LN -> Linear -> GELU -> Linear) -> residual.

    Uses optional layer scale and stochastic depth.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt backbone with optional two-stage head for legacy checkpoints.

    When `use_pretrained_head=True`, a head_norm+adapter stack is created to
    remain compatible with older checkpoints trained on large label spaces.
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 use_pretrained_head=False, pretrained_classes=21841):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.use_pretrained_head = use_pretrained_head
        if use_pretrained_head:
            # Two-stage head for legacy checkpoints
            self.head = nn.Linear(dims[-1], pretrained_classes)
            self.head_norm = nn.LayerNorm(pretrained_classes, eps=1e-6)
            self.adapter = nn.Linear(pretrained_classes, num_classes)
        else:
            self.head = nn.Linear(dims[-1], num_classes)
            self.head_norm = None
            self.adapter = None

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        if self.adapter is not None:
            nn.init.normal_(self.adapter.weight, mean=0.0, std=0.001)
            nn.init.constant_(self.adapter.bias, 0.0)
        

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # Avoid reinitializing adapter if present
            if hasattr(self, 'adapter') and self.adapter is not None and m is self.adapter:
                return
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Run downsampling/stages and return global average pooled features."""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        if self.adapter is not None:
            x = self.head_norm(x)
            x = self.adapter(x)
        return x


def convnext_small(num_classes=1, drop_path_rate=0., layer_scale_init_value=1e-6, 
                   head_init_scale=1., use_pretrained_head=False, **kwargs):
    """Factory for ConvNeXt-S variant sized for small-scale tasks."""
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768],
                     num_classes=num_classes, drop_path_rate=drop_path_rate,
                     layer_scale_init_value=layer_scale_init_value,
                     head_init_scale=head_init_scale, use_pretrained_head=use_pretrained_head, **kwargs)
    return model