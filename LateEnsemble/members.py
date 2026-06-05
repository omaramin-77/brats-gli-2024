"""Member specifications for the LateEnsemble (Pipeline D — Version B).

Each entry tells cache_predictions.py how to instantiate one trained member
and which BraTSDataset modality string to feed it. Order is frozen at the
list position because the PerSubregionStacker indexes weights by member
index — changing the order without retraining a checkpoint silently breaks
predictions.

Membership rationale
--------------------
The committed test_metrics.csv shows that no weighted combination of
unimodal models can reach the naive M5 baseline on TC/ET, because the
unimodal predictions never had the chance to encode cross-modality
interactions. M5 is included as the consensus base so the stacked ensemble
has a path to beat the baseline rather than approach it from below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

# Members import their own modules — keep imports lazy inside builder
# closures so that importing members.py does not pay the cost (or hit
# the failure modes) of loading every backbone at once.


@dataclass(frozen=True)
class MemberSpec:
    short: str            # "m1" — used in cache paths
    member_name: str      # MEMBER_NAME passed to CheckpointManager
    modality: str         # BraTSDataset modality string
    builder: Callable[[], nn.Module]


def _build_m1() -> nn.Module:
    from member1_T1n.model import ResidualUNet3D
    return ResidualUNet3D(in_channels=1, out_channels=3)


def _build_m2() -> nn.Module:
    from member2_T1c.model import AttentionUNet3D
    return AttentionUNet3D(in_channels=1, out_channels=3)


def _build_m3():
    # SwinUNETR's state_dict is shape-tied to img_size; build at PATCH_SIZE
    # so the checkpoint loads, and rely on sliding_window_inference to feed
    # PATCH_SIZE-shaped patches through forward().
    from shared.config import PATCH_SIZE
    from member3_T2w.model import build_model
    return build_model(img_size=PATCH_SIZE, feature_size=24)


def _build_m4() -> nn.Module:
    from member4_T2f.model import T2fSegModel
    return T2fSegModel(in_channels=1, out_channels=3)


def _build_m5() -> nn.Module:
    from member5_multimodal.model import MultimodalUNet3D
    return MultimodalUNet3D(in_channels=4, out_channels=3)


# ----------------------------------------------------------------------
# The canonical, frozen ordering. Do not reorder — the stacker indexes
# weights by position, and a fitted stacker checkpoint will mis-apply
# weights to the wrong members under any rearrangement.
# ----------------------------------------------------------------------
MEMBER_SPECS: list[MemberSpec] = [
    MemberSpec("m1", "M1_T1n_ResUNet",     "t1n",        _build_m1),
    MemberSpec("m2", "M2_T1c_AttUNet",     "t1c",        _build_m2),
    MemberSpec("m3", "M3_T2w_SwinUNETR",   "t2w",        _build_m3),
    MemberSpec("m4", "M4_T2f_resunet",     "t2f",        _build_m4),
    MemberSpec("m5", "M5_Multimodal_4ch",  "multimodal", _build_m5),
]

NUM_MEMBERS = len(MEMBER_SPECS)
SHORT_TO_MODALITY = {s.short: s.modality for s in MEMBER_SPECS}
# Mapping from member short name to the modality key used in the ablation
# JSON. M5 has no per-modality entry (it consumes all four); the stacker
# uses an explicit prior weight for it instead.
SHORT_TO_ABLATION_KEY = {"m1": "t1n", "m2": "t1c", "m3": "t2w", "m4": "t2f"}
