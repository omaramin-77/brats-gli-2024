"""3D Grad-CAM for volumetric segmentation networks.

Members attach this to the last convolutional block of their encoder/decoder
(typically the layer just before the final 1x1x1 classifier convolution). The
implementation mirrors the original Grad-CAM paper, generalised from 2D to 3D
by averaging gradients over the (D, H, W) spatial axes.

Preferred usage (per sub-region)::

    with GradCAM3D(model, target_layer) as cam:
        cam_wt = cam.generate(input_tensor, target_channel="wt")
        cam_tc = cam.generate(input_tensor, target_channel="tc")
        cam_et = cam.generate(input_tensor, target_channel="et")

Each call performs a fresh forward+backward pass scoped to one
output channel. Hooks are released on ``__exit__``.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM3D:
    """Reusable 3D Grad-CAM with explicit hook lifecycle management."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self.last_target_channel = None
        self._param_grad_state: dict = {}

        # Forward hook captures activations on the chosen layer.
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        # full_backward_hook gives us gradients of the loss w.r.t. the layer output.
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _save_activation(self, _module, _inputs, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_channel = "wt",
        target_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Compute a Grad-CAM heatmap aligned with ``input_tensor`` for one
        output sub-region channel.

        ``target_channel`` is either an int index (0/1/2) or a string
        ("wt"/"tc"/"et") mapping to channel 0/1/2 of the model output.

        ``target_mask`` selects which voxels of the chosen channel drive the
        backward pass. If None we use the predicted-positive voxels of that
        channel (sigmoid > 0.5).
        """
        self.model.eval()
        # Backward pass needs grad enabled; flags are flipped on by __enter__
        # and restored by __exit__ so the caller's model state is preserved.
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        # If model returns a tuple (e.g. M1 with deep supervision in training
        # mode), pick the first element (main prediction).
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Resolve target_channel to an int index.
        ch_map = {"wt": 0, "tc": 1, "et": 2}
        if isinstance(target_channel, str):
            if target_channel not in ch_map:
                raise ValueError(
                    f"target_channel must be 'wt'/'tc'/'et' or int, got {target_channel!r}"
                )
            ch = ch_map[target_channel]
        else:
            ch = int(target_channel)
        self.last_target_channel = target_channel

        # Single-channel output? Use it directly (backward compat).
        if output.shape[1] == 1:
            channel_logits = output
        else:
            channel_logits = output[:, ch:ch+1]

        if target_mask is None:
            with torch.no_grad():
                target_mask = (torch.sigmoid(channel_logits) > 0.5).float()
        else:
            target_mask = target_mask.float().to(channel_logits.device)

        # Reduce to a scalar so .backward() has something to differentiate.
        loss = (channel_logits * target_mask).mean()

        self.model.zero_grad(set_to_none=True)
        loss.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire. Check that target_layer is "
                "actually executed during model.forward()."
            )

        # weights: global-average-pool the gradients over spatial dims.
        weights = self._gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))

        # Resample to the input volume's spatial shape.
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        cam = cam.squeeze(0).squeeze(0)
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu().numpy()

    def remove_hooks(self) -> None:
        """Detach hooks. Call this when you are done — leaked hooks pin the
        captured tensors and quickly exhaust GPU memory."""
        try:
            self._fwd_handle.remove()
        except Exception:  # pragma: no cover
            warnings.warn("forward hook was already removed")
        try:
            self._bwd_handle.remove()
        except Exception:  # pragma: no cover
            warnings.warn("backward hook was already removed")
        self._activations = None
        self._gradients = None

    def __enter__(self):
        # Snapshot every parameter's requires_grad flag, then flip all on so
        # the backward pass in generate() can compute gradients regardless
        # of how the model was configured by the caller (eval, frozen, etc.).
        self._param_grad_state = {
            id(p): p.requires_grad for p in self.model.parameters()
        }
        for p in self.model.parameters():
            p.requires_grad_(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        # Restore the caller's requires_grad configuration.
        for p in self.model.parameters():
            if id(p) in self._param_grad_state:
                p.requires_grad_(self._param_grad_state[id(p)])
        self._param_grad_state.clear()
        return False   # do not suppress exceptions
