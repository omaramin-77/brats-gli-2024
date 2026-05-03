"""Member 1 — T1n unimodal segmentation model.

Owner: Member 1
Modality: T1n (native T1-weighted)
Goal: Train a 3D U-Net (or equivalent) that segments whole tumour from T1n alone.
Output: a single-channel logit volume.
"""
from __future__ import annotations

import torch.nn as nn


class T1nSegModel(nn.Module):
    """TODO: implement the encoder-decoder architecture for T1n."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        # TODO: build encoder blocks
        # TODO: build decoder blocks
        # TODO: define final 1x1x1 classification conv
        raise NotImplementedError

    def forward(self, x):
        # TODO: forward pass returning (B, 1, H, W, D) logits
        raise NotImplementedError
