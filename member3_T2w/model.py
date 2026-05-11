"""Member 3 — T2w unimodal segmentation model.

Owner : Member 3
Modality : T2w (T2-weighted)
Architecture : Swin-UNETR (Swin Transformer encoder + CNN decoder)

WHY SWIN-UNETR FOR T2w?
    T2w highlights oedema — large, diffuse regions spanning many centimetres.
    Standard 3x3x3 convolutions have a tiny receptive field and need dozens of
    layers before distant voxels interact. Swin-UNETR processes 3-D patches as
    tokens and computes self-attention inside shifted windows, giving the model
    global spatial context from early layers. This is ideal for capturing the
    full extent of T2w oedema, which is why we expect this model to achieve
    the best whole-tumour Dice among M1-M4.

XAI SUPPORT:
    A forward hook on every WindowAttention module stores raw attention weight
    matrices in self.attention_maps after each forward call. xai_analysis.py
    reads these to produce per-block spatial saliency maps.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class T2wSegModel(nn.Module):
    """
    Swin-UNETR for T2w whole-tumour segmentation.

    Wraps MONAI's SwinUNETR and adds:
      - forward hooks on every WindowAttention module to capture attention maps
      - get_target_layer() for Grad-CAM in xai_analysis.py
      - remove_hooks() to release memory after XAI
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: tuple = (64, 64, 64),
        feature_size: int = 24,
    ):
        super().__init__()

        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,   # gradient checkpointing — saves VRAM
        )

        # Filled by hooks after every forward pass, cleared at start of next
        self.attention_maps: list[torch.Tensor] = []
        self._hooks: list = []
        self._register_attention_hooks()

    def _register_attention_hooks(self) -> None:
        try:
            from monai.networks.nets.swin_unetr import WindowAttention  # type: ignore
        except ImportError:
            return
        for module in self.model.modules():
            if isinstance(module, WindowAttention):
                self._hooks.append(
                    module.register_forward_hook(self._attn_hook)
                )

    def _attn_hook(self, module, inputs, output) -> None:
        if isinstance(output, tuple) and len(output) == 2:
            self.attention_maps.append(output[1].detach().cpu())

    def remove_hooks(self) -> None:
        """Call after Grad-CAM / XAI — prevents GPU memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_target_layer(self) -> nn.Module:
        """Last decoder block before classifier — used as Grad-CAM target."""
        for attr in ("decoder1", "decoder2"):
            layer = getattr(self.model, attr, None)
            if layer is not None:
                return layer
        children = list(self.model.children())
        return children[-2] if len(children) >= 2 else children[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.attention_maps.clear()
        return self.model(x)


def build_model(img_size: tuple = (64, 64, 64), feature_size: int = 24) -> T2wSegModel:
    """
    Factory used by train.py, evaluate.py, and xai_analysis.py.

    img_size    : must match PATCH_SIZE from shared/config.py
    feature_size: 24 = safe for 6 GB VRAM (RTX 3050)
                  48 = standard, needs ~10 GB
                  96 = best quality, needs ~24 GB
    """
    return T2wSegModel(
        in_channels=1,
        out_channels=1,       # binary sigmoid — one channel
        img_size=img_size,
        feature_size=feature_size,
    )


# ---------------------------------------------------------------------------
# Smoke-test:  python member3_T2w/model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(img_size=(64, 64, 64)).to(device)
    x = torch.randn(1, 1, 64, 64, 64, device=device)

    with torch.no_grad():
        out = model(x)

    total = sum(p.numel() for p in model.parameters())
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(out.shape)}")
    print(f"Params : {total/1e6:.1f} M")
    print(f"Hooks  : {len(model._hooks)}")
    print("Smoke-test PASSED ✓")
