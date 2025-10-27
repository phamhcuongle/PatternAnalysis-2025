from typing import List
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pw = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # move to channels-last for MLP
        x = x.permute(0, 2, 3, 1)
        # apply layernorm on channels (as channels_last) via functional
        x = nn.functional.layer_norm(x, x.shape[-1:], None, None, 1e-6)
        x = self.pw(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)
        self.norm = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        x = self.proj(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=2, depths: List[int] = None, dims: List[int] = None,
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        if depths is None:
            depths = [3, 4, 18, 3]  # ConvNeXt-B
        if dims is None:
            dims = [128, 256, 512, 1024]

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            down = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(down)

        # stochastic depth decay rule
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        cur = 0
        for i_stage in range(4):
            blocks = []
            for j in range(depths[i_stage]):
                blocks.append(ConvNeXtBlock(dims[i_stage], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value))
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i_stage]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for idx, down in enumerate(self.downsample_layers):
            x = down(x)
            x = self.stages[idx](x)
        # global pool
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x


def convnext_b(num_classes=2, in_chans=3, pretrained_path=None, device='cpu'):
    model = ConvNeXt(in_chans=in_chans, num_classes=num_classes)
    if pretrained_path:
        sd = torch.load(pretrained_path, map_location=device)
        # try load flexibly
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            # some checkpoints wrap weights under 'model' key
            if isinstance(sd, dict) and 'model' in sd:
                model.load_state_dict(sd['model'], strict=False)
            else:
                # best effort: filter keys that match
                own_state = model.state_dict()
                filtered = {k: v for k, v in sd.items() if k in own_state and own_state[k].shape == v.shape}
                own_state.update(filtered)
                model.load_state_dict(own_state)
    return model