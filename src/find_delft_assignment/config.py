from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from find_delft_assignment.model.pretrained import PretrainedDDPMBundle


@dataclass(frozen=True)
class SamplingConfig:
    """Inference hyperparameters aligned with a concrete DDPM checkpoint."""

    num_train_timesteps: int = 1000
    num_inference_steps: int = 1000
    eta: float = 0.0
    seed: int = 0
    image_size: int = 32
    in_channels: int = 3

    @classmethod
    def from_pretrained(
        cls,
        bundle: PretrainedDDPMBundle,
        *,
        num_inference_steps: int | None = None,
        eta: float = 0.0,
        seed: int = 0,
    ) -> SamplingConfig:
        """Build a config from a Hub bundle so spatial size and train steps match the checkpoint."""
        n_train = int(bundle.ddpm_scheduler.config.num_train_timesteps)
        steps = n_train if num_inference_steps is None else int(num_inference_steps)
        return cls(
            num_train_timesteps=n_train,
            num_inference_steps=steps,
            eta=float(eta),
            seed=int(seed),
            image_size=int(bundle.image_size),
            in_channels=int(bundle.in_channels),
        )
