"""Basic neural-network models for the deep solvers."""

import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    """A small tanh multilayer perceptron: in_dim -> hidden -> ... -> out_dim."""

    def __init__(self, in_dim, out_dim, hidden=32, depth=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
