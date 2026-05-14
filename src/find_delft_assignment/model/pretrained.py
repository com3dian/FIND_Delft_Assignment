from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

DEFAULT_PRETRAINED_MODEL_ID = "google/ddpm-cifar10-32"  # 32×32 CIFAR-10 DDPM; strong FID for small footprint.


@dataclass(frozen=True)
class PretrainedDDPMBundle:
    """UNet plus DDPM schedule from the same Hub snapshot (weights and noise schedule stay aligned)."""

    model_id: str
    unet: nn.Module
    ddpm_scheduler: Any
    image_size: int
    in_channels: int


def load_pretrained_ddpm(
    model_id: str = DEFAULT_PRETRAINED_MODEL_ID,
    *,
    torch_dtype: torch.dtype | None = None,
    use_safetensors: bool = False,
) -> PretrainedDDPMBundle:
    """Load the pretrained UNet and DDPM scheduler from the Hugging Face Hub.

    The ``google/ddpm-cifar10-32`` checkpoint ships a small custom pipeline module; we enable
    ``trust_remote_code`` only for that Hub entry so weights and architecture resolve correctly.

    ``use_safetensors`` defaults to ``False`` because several diffusers + Hub combinations raise
    when forcing safetensors on this flat layout repo even though ``.safetensors`` exists; PyTorch
    ``.bin`` weights load reliably.
    """
    try:
        from diffusers import DDPMPipeline
    except ImportError as exc:  # pragma: no cover - import guard for optional stack
        raise ImportError(
            "diffusers is required to load the pretrained model. Install dependencies: pip install -e ."
        ) from exc

    kwargs: dict[str, Any] = {"use_safetensors": use_safetensors}
    if model_id == DEFAULT_PRETRAINED_MODEL_ID:
        kwargs["trust_remote_code"] = True

    try:
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    except OSError as exc:
        if use_safetensors and "safetensors" in str(exc).lower():
            kwargs["use_safetensors"] = False
            pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
        else:
            raise
    unet = pipe.unet
    scheduler = pipe.scheduler
    sample_size = int(getattr(unet.config, "sample_size", 32))
    in_ch = int(getattr(unet.config, "in_channels", 3))
    return PretrainedDDPMBundle(
        model_id=model_id,
        unet=unet,
        ddpm_scheduler=scheduler,
        image_size=sample_size,
        in_channels=in_ch,
    )
