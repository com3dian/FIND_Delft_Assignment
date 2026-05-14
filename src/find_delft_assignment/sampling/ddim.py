from __future__ import annotations

from typing import Any

import torch
from torch import nn

from find_delft_assignment.config import SamplingConfig
from find_delft_assignment.sampling.common import unet_predict_noise
from find_delft_assignment.sampling.scheduler import DDIMSchedulerAdapter


def _ddim_scheduler_from_ddpm(ddpm_scheduler: Any) -> DDIMSchedulerAdapter:
    return DDIMSchedulerAdapter(ddpm_scheduler)


def ddim_sample(
    unet: nn.Module,
    ddpm_scheduler: Any,
    config: SamplingConfig,
    *,
    batch_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    DDIM sampling (Song et al., 2020) with optional step reduction.

    Uses :class:`DDIMSchedulerAdapter` (same update as diffusers ``DDIMScheduler`` with
    ``clip_sample=False`` on the reference) so ``alphas_cumprod`` matches the loaded DDPM.
    """
    unet = unet.to(device)
    unet.eval()
    dtype = next(unet.parameters()).dtype
    h = w = config.image_size

    scheduler = _ddim_scheduler_from_ddpm(ddpm_scheduler)
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

            timesteps = tqdm(timesteps, desc="DDIM", leave=False)
        except ImportError:  # pragma: no cover
            pass

    eta = float(config.eta)
    with torch.inference_mode():
        for t in timesteps:
            noise_pred = unet_predict_noise(unet, sample, t)
            ti = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            step_out = scheduler.step(
                noise_pred,
                ti,
                sample,
                eta=eta,
                generator=generator,
            )
            sample = step_out.prev_sample
    return sample
