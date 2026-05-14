from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> Any:
    """Load a raw torch checkpoint (dict or tensor). Wire `map_location` for CPU/CUDA."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location)
