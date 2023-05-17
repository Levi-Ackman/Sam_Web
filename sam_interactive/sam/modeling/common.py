## This is the reproducity work by Levi-Ack from UESTC at 2023/5/8
## The basic reference comes from  https://github.com/facebookresearch/segment-anything/tree/main/segment_anything/modeling
## you may check out the ori-paper for a detailed look :https://ai.facebook.com/research/publications/segment-anything/

import torch
import torch.nn as nn
from typing import Type


## classical MLP block, input => expansion => ori_dim
## typically used in transformers's feedforward layers
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    
'''
function:
    Given the input tensor x, compute its mean u and variance s along the channel dimension.
    Normalize the input based on u and s: subtract the mean from x, divide by the variance plus a small constant term.
    Multiply the standardized result by a learnable scale factor (i.e., self.weight) and add a learnable bias factor (i.e., self.bias).
Important:arguments:
    x_i is the i-th element of the input tensor x,
    μ is the mean of x along the current channel,
    σ is the variance of x along the current channel,
    ϵ is a small constant (default to 10^-6) to prevent division by zero,
    γ and β are the learnable scale and shift factors.
'''
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
