"""Member 5 — Multimodal (T1c + T1n + T2f + T2w) segmentation model.

Owner: Member 5
Modality: all four modalities stacked as input channels (in this order:
t1c, t1n, t2f, t2w — matches shared.preprocessing.preprocess_patient_multimodal).
Goal: Train a single 3D segmentation network that consumes the full
multimodal tensor and serves as the project's main reference baseline.
"""
from __future__ import annotations

import torch.nn as nn


class MultimodalSegModel(nn.Module):
    """TODO: implement multimodal encoder-decoder.

    Input shape:  (B, 4, H, W, D)  — channel order (t1c, t1n, t2f, t2w)
    Output shape: (B, 3, H, W, D)  — WT/TC/ET logits (BraTS protocol)
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        # TODO: encoder/decoder/classifier
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
