"""Member 3 — T2w unimodal segmentation model.

Owner: Member 3
Modality: T2w (T2-weighted)
Goal: Segment whole tumour from T2w. T2w highlights oedema/non-enhancing
disease so this baseline complements the T1-based members.
"""
from __future__ import annotations

import torch.nn as nn


class T2wSegModel(nn.Module):
    """TODO: implement encoder-decoder for T2w."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()
        # TODO: encoder/decoder/classifier
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
