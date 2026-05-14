from __future__ import annotations

import inspect

import torch
from torch import nn


def _unet_accepts_kwarg(unet: nn.Module, name: str) -> bool:
    return name in inspect.signature(unet.forward).parameters


def unet_predict_noise(
    unet: nn.Module,
    sample: torch.Tensor,
    timestep: torch.Tensor | int
    ) -> torch.Tensor:
    """Run the score network and return epsilon prediction (``prediction_type='epsilon'``)."""
    if isinstance(timestep, int):
        t = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
    else:
        t = timestep.to(device=sample.device, dtype=torch.long)
        if t.ndim == 0:
            t = t.expand(sample.shape[0])

    kwargs: dict[str, bool] = {}
    if _unet_accepts_kwarg(unet, "return_dict"):
        kwargs["return_dict"] = False

    out = unet(sample, t, **kwargs)
    if isinstance(out, tuple):
        return out[0]
    return out.sample
