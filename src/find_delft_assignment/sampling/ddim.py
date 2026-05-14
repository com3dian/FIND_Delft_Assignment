from __future__ import annotations

from typing import Any

from find_delft_assignment.config import SamplingConfig


def ddim_sample_stub(
    model: Any,
    config: SamplingConfig,
) -> Any:
    """DDIM sampling with optional step reduction; implement for your noise schedule."""
    raise NotImplementedError(
        "Implement DDIM updates (eta, sub-sampled timesteps) on top of the same DDPM model."
    )
