"""Compact U-Net for floor-plan segmentation (PyTorch).

Classes (index): 0 background, 1 wall, 2 door, 3 window.
Export contract matches perception.learned.OnnxWallModel: input float32 [1,1,H,W]
(0 = ink, 1 = paper), output [1,K,H,W] class scores -> use wall_model_channel=1.
"""
from __future__ import annotations

import torch
from torch import nn

CLASSES = ("background", "wall", "door", "window")


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = len(CLASSES), base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(in_ch, chs[0])] + [_block(chs[i], chs[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_block(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


class Softmaxed(nn.Module):
    """Wrapper baking softmax into the exported graph (OnnxWallModel expects scores/probs)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)
