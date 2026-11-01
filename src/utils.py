import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set random seeds used by FlowCast's Python, NumPy, and PyTorch stages."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
