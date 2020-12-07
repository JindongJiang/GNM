from typing import List, Tuple, Dict, Callable, Any
import torch
from torch import nn


class StackConvNorm(nn.Module):
    def __init__(self,
                 dim_inp: int,
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 groupings: List[int],
                 norm_act_final: bool,
                 activation: Callable = nn.CELU):
        super(StackConvNorm, self).__init__()

        layers = []

        dim_prev = dim_inp

        for i, (f, k, s) in enumerate(zip(filters, kernel_sizes, strides)):
            if s == 0:
                layers.append(nn.Conv2d(dim_prev, f, k, 1, 0))
            else:
                layers.append(nn.Conv2d(dim_prev, f, k, s, (k - 1) // 2))
            if i == len(filters) - 1 and norm_act_final == False:
                break
            layers.append(activation())
            layers.append(nn.GroupNorm(groupings[i], f))
            # layers.append(nn.BatchNorm2d(f))
            dim_prev = f

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        return x


class StackSubPixelNorm(nn.Module):
    def __init__(self,
                 dim_inp: int,
                 filters: List[int],
                 kernel_sizes: List[int],
                 upscale: List[int],
                 groupings: List[int],
                 norm_act_final: bool,
                 activation: Callable = nn.CELU):
        super(StackSubPixelNorm, self).__init__()

        layers = []

        dim_prev = dim_inp

        for i, (f, k, u) in enumerate(zip(filters, kernel_sizes, upscale)):
            if u == 1:
                layers.append(nn.Conv2d(dim_prev, f, k, 1, (k - 1) // 2))
            else:
                layers.append(nn.Conv2d(dim_prev, f * u ** 2, k, 1, (k - 1) // 2))
                layers.append(nn.PixelShuffle(u))
            if i == len(filters) - 1 and norm_act_final == False:
                break
            layers.append(activation())
            layers.append(nn.GroupNorm(groupings[i], f))
            dim_prev = f

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        return x


class StackMLP(nn.Module):
    def __init__(self,
                 dim_inp: int,
                 filters: List[int],
                 norm_act_final: bool,
                 activation: Callable = nn.CELU,
                 phase_layer_norm: bool = True):
        super(StackMLP, self).__init__()

        layers = []

        dim_prev = dim_inp

        for i, f in enumerate(filters):
            layers.append(nn.Linear(dim_prev, f))
            if i == len(filters) - 1 and norm_act_final == False:
                break
            layers.append(activation())
            if phase_layer_norm:
                layers.append(nn.LayerNorm(f))
            dim_prev = f

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)

        return x


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_cell=4):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=self.input_dim + hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=True)

        self.register_parameter('h_0', torch.nn.Parameter(torch.zeros(1, self.hidden_dim, num_cell, num_cell),
                                                          requires_grad=True))
        self.register_parameter('c_0', torch.nn.Parameter(torch.zeros(1, self.hidden_dim, num_cell, num_cell),
                                                          requires_grad=True))

    def forward(self, x, h_c):
        h_cur, c_cur = h_c

        conv_inp = torch.cat([x, h_cur], dim=1)

        i, f, o, c = self.conv(conv_inp).split(self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c = torch.tanh(c)
        o = torch.sigmoid(o)

        c_next = f * c_cur + i * c
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return self.h_0.expand(batch_size, -1, -1, -1), \
               self.c_0.expand(batch_size, -1, -1, -1)
