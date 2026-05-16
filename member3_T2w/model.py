# member3_T2w/model.py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from shared.config import TARGET_CHANNELS


class M3T2wSwinUNETR(nn.Module):
    """
    Member 3 model: T2w Swin-UNETR.

    Output:
        (B, 3, H, W, D) logits for WT / TC / ET.

    Latent fusion:
        forward_features(x) returns bottleneck features projected to
        (B, 256, H/16, W/16, D/16).
    """

    def __init__(
        self,
        img_size=(64, 64, 64),
        in_channels: int = 1,
        out_channels: int = TARGET_CHANNELS,
        feature_size: int = 24,
    ):
        super().__init__()

        # MONAI versions differ: some require img_size, some do not.
        try:
            self.model = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                use_checkpoint=True,
            )
        except TypeError:
            self.model = SwinUNETR(
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                use_checkpoint=True,
            )

        self.feature_size = feature_size
        self.feature_projector = nn.Conv3d(feature_size * 16, 256, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract Swin bottleneck features for M5 latent fusion.
        """
        try:
            hidden_states = self.model.swinViT(x, self.model.normalize)
        except TypeError:
            hidden_states = self.model.swinViT(x)

        bottleneck = hidden_states[-1]

        # Some MONAI versions return channel-last: (B, D, H, W, C)
        if bottleneck.shape[1] != self.feature_size * 16:
            bottleneck = bottleneck.permute(0, 4, 1, 2, 3).contiguous()

        return self.feature_projector(bottleneck)

    def get_target_layer(self) -> nn.Module:
        """
        Target layer for Grad-CAM.
        """
        if hasattr(self.model, "decoder5"):
            return self.model.decoder5
        if hasattr(self.model, "encoder10"):
            return self.model.encoder10
        return list(self.model.children())[-2]


def build_model(img_size=(64, 64, 64), feature_size: int = 24) -> M3T2wSwinUNETR:
    return M3T2wSwinUNETR(
        img_size=img_size,
        in_channels=1,
        out_channels=3,
        feature_size=feature_size,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(img_size=(64, 64, 64), feature_size=24).to(device)
    model.eval()

    x = torch.randn(1, 1, 64, 64, 64, device=device)

    with torch.no_grad():
        y = model(x)
        f = model.forward_features(x)

    print("Input shape   :", tuple(x.shape))
    print("Output shape  :", tuple(y.shape))
    print("Feature shape :", tuple(f.shape))

    assert tuple(y.shape) == (1, 3, 64, 64, 64), f"Wrong output shape: {y.shape}"
    assert f.shape[1] == 256, f"Wrong feature channels: {f.shape}"

    print("Smoke-test PASSED")