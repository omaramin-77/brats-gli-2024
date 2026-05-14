"""Member 4 — T2f (FLAIR) unimodal segmentation model.

Owner: Member 4
Modality: T2-FLAIR
Goal: Segment whole tumour from FLAIR. FLAIR is the canonical sequence for
delineating peritumoural oedema, so this baseline often gives the best
single-modality whole-tumour Dice.
"""
from __future__ import annotations

import torch.nn as nn


class T2fSegModel(nn.Module):
    """TODO: implement encoder-decoder for T2f/FLAIR."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()
        # TODO: encoder/decoder/classifier
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
