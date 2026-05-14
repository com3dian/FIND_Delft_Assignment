from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TimedResult:
    """Wall-clock timing for a forward-only sampling call."""

    seconds: float
    batch_size: int
    num_inference_steps: int

    @property
    def seconds_per_image(self) -> float:
        return self.seconds / max(self.batch_size, 1)

    @property
    def steps_per_second(self) -> float:
        total_steps = self.num_inference_steps * self.batch_size
        return total_steps / self.seconds if self.seconds > 0 else float("inf")


def time_callable(fn, *, rounds: int = 1) -> float:
    """Return mean wall time in seconds for ``fn()`` over ``rounds`` runs (sync CUDA if used)."""
    times: list[float] = []
    for _ in range(rounds):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def pixel_diversity(samples: torch.Tensor) -> dict[str, float]:
    """Lightweight diversity proxies on a small batch in ``[-1, 1]`` (not FID).

    - ``mean_pairwise_l2``: mean L2 distance between flattened images (higher → more spread).
    - ``channel_std_mean``: mean spatial std per channel (variation within images).
    """
    b = samples.shape[0]
    flat = samples.reshape(b, -1).float()
    if b < 2:
        return {"mean_pairwise_l2": 0.0, "channel_std_mean": float(samples.float().std())}

    dist = torch.cdist(flat, flat, p=2)
    upper = torch.triu(torch.ones_like(dist), diagonal=1).bool()
    mean_pw = float(dist.masked_select(upper).mean().item())
    ch_std = float(samples.float().std(dim=(0, 2, 3)).mean().item())
    return {"mean_pairwise_l2": mean_pw, "channel_std_mean": ch_std}
