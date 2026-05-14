from __future__ import annotations

from pathlib import Path
from typing import Any


def save_image_grid(images: Any, path: str | Path) -> None:
    """Save a tensor or array batch as a grid image under ``outputs/``."""
    raise NotImplementedError("Use torchvision.utils.save_image or Pillow to write grids.")
