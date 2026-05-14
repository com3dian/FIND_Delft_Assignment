from __future__ import annotations

from pathlib import Path

import torch


def save_image_grid(images: torch.Tensor, path: str | Path, *, nrow: int = 8, padding: int = 2) -> None:
    """Save a batch ``(N, C, H, W)`` in ``[-1, 1]`` as a single grid PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from torchvision.utils import make_grid, save_image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torchvision is required for save_image_grid (pip install torchvision).") from exc

    grid = make_grid(images.detach().cpu().float(), nrow=nrow, padding=padding, normalize=True, value_range=(-1.0, 1.0))
    save_image(grid, path)
