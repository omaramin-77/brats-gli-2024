"""Deterministic seeding for the whole pipeline.

Every train.py / evaluate.py must call set_global_seed() as its first executable
line so that data shuffling, augmentation, and weight initialisation are all
reproducible across machines.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Seed every random source we know about.

    Note: cuDNN benchmark must be False — when True, cuDNN picks the fastest
    convolution algorithm at runtime, which is non-deterministic across runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] Global seed set to {seed}")
