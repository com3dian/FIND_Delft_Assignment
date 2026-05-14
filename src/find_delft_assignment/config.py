from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingConfig:
    """Defaults for inference; extend when you wire a concrete DDPM."""

    num_train_timesteps: int = 1000
    num_inference_steps: int = 1000
    eta: float = 0.0
    seed: int = 0
