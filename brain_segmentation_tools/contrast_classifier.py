"""Self-contained inference module wrapping preprocessing + the trained CNN."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2),
    )


class LightweightCNN3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 2,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        blocks = []
        prev = in_channels
        for ch in channels:
            blocks.append(_conv_block(prev, ch))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev, prev // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(prev // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


_CSF_IDS = {4, 5, 14, 15, 24, 31, 43, 44, 63, 72, 213, 221}
_WM_IDS = {2, 7, 16, 28, 41, 46, 60, 85, 192, 251, 252, 253, 254, 255}
_SUBCORTICAL_GM_IDS = {8, 10, 11, 12, 13, 17, 18, 26, 47, 49, 50, 51, 52, 53, 54, 58}
_CORTEX_IDS = {3, 42}

NUM_TISSUE_CLASSES = 4


class ContrastClassificationModel(nn.Module):
    """Takes an unpooled image + SynthSeg volume and returns P(post-contrast)."""

    def __init__(self, checkpoint_path: str | Path, pool_factor: int = 2) -> None:
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # self.image_size = tuple(checkpoint["config"]["image_size"])
        self.pool_factor = pool_factor
        self.net = LightweightCNN3D(in_channels=2, num_classes=2)
        self.net.load_state_dict(checkpoint["model_state"])
        self.net.eval()

        self.register_buffer("_gm_ids", torch.tensor(sorted(_SUBCORTICAL_GM_IDS | _CORTEX_IDS)))
        self.register_buffer("_csf_ids", torch.tensor(sorted(_CSF_IDS)))
        self.register_buffer("_wm_ids", torch.tensor(sorted(_WM_IDS)))

    def _to_tissue(self, seg: torch.Tensor) -> torch.Tensor:
        tissue = torch.zeros_like(seg)
        gm_ids = cast(torch.Tensor, self._gm_ids)
        csf_ids = cast(torch.Tensor, self._csf_ids)
        wm_ids = cast(torch.Tensor, self._wm_ids)
        assert gm_ids is not None
        assert csf_ids is not None
        assert wm_ids is not None
        seg_labels = seg.to(dtype=torch.int64)
        gm_mask = torch.zeros_like(seg_labels, dtype=torch.bool)
        for label in gm_ids:
            gm_mask |= seg_labels == label
        csf_mask = torch.zeros_like(seg_labels, dtype=torch.bool)
        for label in csf_ids:
            csf_mask |= seg_labels == label
        wm_mask = torch.zeros_like(seg_labels, dtype=torch.bool)
        for label in wm_ids:
            wm_mask |= seg_labels == label
        tissue[gm_mask | (seg >= 1000)] = 2.0
        tissue[csf_mask] = 1.0
        tissue[wm_mask] = 3.0
        return tissue / float(NUM_TISSUE_CLASSES - 1)

    def forward(self, image: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """`image`/`seg`: (B, 1, H, W, D) at native (unpooled) resolution."""
        image = image.float()
        seg = seg.float()
        # The network downsamples by pool_factor * 2**num_blocks; pad so no spatial
        # axis collapses to zero for thin/small crops.
        min_size = self.pool_factor * 2 ** len(self.net.features)
        pad = []
        for dim in reversed(image.shape[2:]):
            deficit = max(0, min_size - dim)
            pad.extend([deficit // 2, deficit - deficit // 2])
        if any(pad):
            image = F.pad(image, pad)
            seg = F.pad(seg, pad)

        image = F.max_pool3d(image, self.pool_factor)
        seg = F.max_pool3d(seg, self.pool_factor)

        # image = F.interpolate(image, size=self.image_size, mode="trilinear", align_corners=False)
        # seg = F.interpolate(seg, size=self.image_size, mode="nearest")

        # nonzero = image != 0
        # mean = (image * nonzero).sum(dim=(1, 2, 3, 4), keepdim=True) / nonzero.sum(dim=(1, 2, 3, 4), keepdim=True)
        # std = (((image - mean) * nonzero) ** 2).sum(dim=(1, 2, 3, 4), keepdim=True).div(
        #     nonzero.sum(dim=(1, 2, 3, 4), keepdim=True)
        # ).sqrt()
        # image = torch.where(nonzero, (image - mean) / (std + 1e-8), image)

        logits = self.net(torch.cat([image, self._to_tissue(seg)], dim=1))
        return torch.softmax(logits, dim=1)[:, 1]
