from __future__ import annotations

from typing import Any

import torch
from torch import nn

from find_delft_assignment.config import SamplingConfig
from find_delft_assignment.sampling.common import unet_predict_noise


def ddpm_sample(
    unet: nn.Module,
    scheduler: Any,
    config: SamplingConfig,
    *,
    batch_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Ancestral DDPM sampling: full Markov reverse chain using the pretrained noise schedule.

    Per-step updates follow DDPMScheduler.step (Ho et al., 2020), driven by the scheduler loaded
    with the checkpoint so betas, variance_type, and clipping match training.
    """
    unet = unet.to(device)
    unet.eval()
    dtype = next(unet.parameters()).dtype
    h = w = config.image_size

    scheduler.set_timesteps(config.num_inference_steps, device=device)
    sample = torch.randn(
        (batch_size, config.in_channels, h, w),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    timesteps = scheduler.timesteps
    if show_progress:
        try:
            from tqdm.auto import tqdm

            timesteps = tqdm(timesteps, desc="DDPM", leave=False)
        except ImportError:  # pragma: no cover
            pass

    with torch.inference_mode():
        for t in timesteps:
            noise_pred = unet_predict_noise(unet, sample, t)
            step_out = scheduler.step(noise_pred, t, sample, generator=generator)
            sample = step_out.prev_sample
    return sample
