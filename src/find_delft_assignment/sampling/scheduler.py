from __future__ import annotations

from typing import Any

import torch
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor


class DDIMSchedulerAdapter:
    def __init__(self, DDPMScheduler: Any):
        self.DDPMScheduler = DDPMScheduler
        self.steps = int(self.DDPMScheduler.config.num_train_timesteps)

    def set_timesteps(self, num_steps: int, device: torch.device | str):
        self.inference_steps = num_steps
        self.stride = self.steps // self.inference_steps  # e.g. 1000 // 45 == 22
        self.timesteps = torch.arange(self.inference_steps - 1, -1, -1) * self.stride
        self.timesteps = self.timesteps.to(device)

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> DDIMSchedulerOutput:
        """
        Step the DDIM reverse proess for one step.
        Ref:
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py.
        """
        acp = self.DDPMScheduler.alphas_cumprod.to(
            device=sample.device, dtype=torch.float32
        )
        prev_idx = timestep - self.stride

        alpha_prod = acp[timestep]
        beta_prod = 1.0 - alpha_prod

        if prev_idx < 0:
            # Final step: no previous noise level in the chain — match diffusers (ᾱ = 1).
            alpha_prod_prev = torch.ones_like(alpha_prod)
            beta_prod_prev = torch.zeros_like(beta_prod)
            variance = torch.zeros_like(alpha_prod)
        else:
            alpha_prod_prev = acp[prev_idx]
            beta_prod_prev = 1.0 - alpha_prod_prev
            variance = (beta_prod_prev / beta_prod) * (1.0 - alpha_prod / alpha_prod_prev)

        # Predicted x_0 from ε (prediction_type == "epsilon").
        pred_original_sample = (sample - beta_prod**0.5 * noise_pred) / alpha_prod**0.5
        pred_epsilon = noise_pred

        std_dev = eta * variance.clamp(min=0.0) ** 0.5

        # Match diffusers: sqrt coefficient uses σ_t^2 = std_dev^2, not the raw variance (see DDIM eq. 12).
        pred_sample_direction = (
            (1.0 - alpha_prod_prev - std_dev**2).clamp(min=0.0) ** 0.5 * pred_epsilon
        )

        prev_sample = alpha_prod_prev**0.5 * pred_original_sample + pred_sample_direction

        variance_noise = randn_tensor(
            sample.shape, generator=generator, device=sample.device, dtype=sample.dtype
        )
        prev_sample = prev_sample + std_dev * variance_noise

        return DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
