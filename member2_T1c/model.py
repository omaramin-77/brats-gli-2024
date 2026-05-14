"""Member 2 — T1c unimodal segmentation model.

Owner: Member 2
Modality: T1c (contrast-enhanced T1)
Goal: Segment whole tumour from T1c alone. Contrast-enhanced T1 is most
informative for the enhancing-tumour subregion, so this baseline is expected
to outperform T1n.
"""
from __future__ import annotations

import torch.nn as nn


class T1cSegModel(nn.Module):
    """TODO: implement encoder-decoder for T1c."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()
        # TODO: encoder
        # TODO: decoder
        # TODO: final classification conv
        raise NotImplementedError

    def forward(self, x):
        # TODO: forward pass
        raise NotImplementedError
