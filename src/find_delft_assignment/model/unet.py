from __future__ import annotations

import torch.nn as nn


class UNetStub(nn.Module):
    """Placeholder score network; replace with your pretrained DDPM architecture."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, t):
        raise NotImplementedError("Wire your pretrained UNet here.")
