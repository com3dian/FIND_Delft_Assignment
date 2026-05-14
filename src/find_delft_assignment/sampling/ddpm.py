from __future__ import annotations

from typing import Any

from find_delft_assignment.config import SamplingConfig


def ddpm_sample_stub(
    model: Any,
    config: SamplingConfig,
) -> Any:
    """Ancestral DDPM sampling; implement using your schedule and checkpoint."""
    raise NotImplementedError(
        "Implement DDPM sampling (full Markov chain) for your pretrained model."
    )
