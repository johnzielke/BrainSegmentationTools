from __future__ import annotations

from collections import OrderedDict
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _QCEncoderLevel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding="same")
        self.conv1 = nn.Conv3d(
            out_channels, out_channels, kernel_size=5, padding="same"
        )
        self.expand_conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=5, padding="same"
        )
        # Keras BatchNorm defaults to epsilon=1e-3.
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.conv0(x))
        x = self.conv1(x)
        identity = F.relu(self.expand_conv(identity))
        x = F.relu(x + identity)
        x = self.batch_norm(x)
        return x


class QCSynthSegRegressor(nn.Module):
    TARGET_SHAPE = 224

    def __init__(self, *, labels_segmentation: list[int], labels_qc: list[int]):
        super().__init__()
        if len(labels_segmentation) != len(labels_qc):
            raise ValueError(
                "labels_segmentation and labels_qc must have the same length"
            )
        self.n_labels_qc = int(max(labels_qc)) + 1

        self.encoder_levels = nn.ModuleList(
            [
                _QCEncoderLevel(self.n_labels_qc, 24),
                _QCEncoderLevel(24, 48),
                _QCEncoderLevel(48, 96),
                _QCEncoderLevel(96, 192),
            ]
        )
        self.level_pools = nn.ModuleList(
            [
                nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
                nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
                nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            ]
        )
        self.final_pool = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.final_conv_0 = nn.Conv3d(
            192, self.n_labels_qc, kernel_size=5, padding="same"
        )
        self.final_conv_1 = nn.Conv3d(
            self.n_labels_qc, self.n_labels_qc, kernel_size=5, padding="same"
        )

        seg_idx_to_label = torch.as_tensor(labels_segmentation, dtype=torch.long)
        self.register_buffer("seg_idx_to_label", seg_idx_to_label)

        lut = torch.zeros(int(seg_idx_to_label.max().item()) + 1, dtype=torch.long)
        for seg_label, qc_label in zip(labels_segmentation, labels_qc, strict=True):
            lut[int(seg_label)] = int(qc_label)
        self.register_buffer("seg_label_to_qc_label", lut)

    def _make_shape(self, seg_idx: torch.Tensor) -> torch.Tensor:
        # Mirrors FreeSurfer's MakeShape(224) used by SynthSeg QC.
        output = []
        shape_target = torch.full(
            (3,), self.TARGET_SHAPE, device=seg_idx.device, dtype=torch.long
        )
        zero = torch.zeros(3, device=seg_idx.device, dtype=torch.long)
        for b in range(seg_idx.shape[0]):
            x = seg_idx[b]
            shape = torch.as_tensor(x.shape, device=seg_idx.device, dtype=torch.long)
            mask = (x != 0) & (x != 24)
            indices = torch.nonzero(mask, as_tuple=False)

            if indices.numel() == 0:
                min_idx = zero
                max_idx = torch.minimum(shape, shape_target)
            else:
                min_idx = torch.maximum(indices.min(dim=0).values.to(torch.long), zero)
                max_idx = torch.minimum(
                    indices.max(dim=0).values.to(torch.long) + 1, shape
                )

            intermediate_shape = max_idx - min_idx
            delta = shape_target - intermediate_shape
            min_idx = min_idx - torch.ceil(delta.float() / 2.0).to(torch.long)
            max_idx = max_idx + torch.floor(delta.float() / 2.0).to(torch.long)

            tmp_min_idx = torch.maximum(min_idx, zero)
            tmp_max_idx = torch.minimum(max_idx, shape)

            x = x[
                tmp_min_idx[0] : tmp_max_idx[0],
                tmp_min_idx[1] : tmp_max_idx[1],
                tmp_min_idx[2] : tmp_max_idx[2],
            ]

            min_padding = torch.abs(torch.minimum(min_idx, zero))
            max_padding = torch.maximum(max_idx - shape, zero)
            if (min_padding > 0).any() or (max_padding > 0).any():
                x = F.pad(
                    x,
                    (
                        int(min_padding[2]),
                        int(max_padding[2]),
                        int(min_padding[1]),
                        int(max_padding[1]),
                        int(min_padding[0]),
                        int(max_padding[0]),
                    ),
                )
            output.append(x)
        return torch.stack(output, dim=0)

    def _prepare_qc_input_from_segmentation(
        self, segmentation: torch.Tensor
    ) -> torch.Tensor:
        seg_idx = torch.argmax(segmentation, dim=1)
        seg_idx = self._make_shape(seg_idx)
        seg_idx_to_label = cast(torch.Tensor, self.seg_idx_to_label)
        seg_label_to_qc_label = cast(torch.Tensor, self.seg_label_to_qc_label)
        seg_labels = seg_idx_to_label[seg_idx]
        qc_labels = seg_label_to_qc_label[seg_labels]
        qc_input = F.one_hot(qc_labels, num_classes=self.n_labels_qc).permute(
            0, 4, 1, 2, 3
        )
        return qc_input.to(dtype=segmentation.dtype)

    def predict_scores_from_segmentation(
        self, segmentation: torch.Tensor
    ) -> torch.Tensor:
        return self(self._prepare_qc_input_from_segmentation(segmentation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for level, encoder in enumerate(self.encoder_levels):
            x = encoder(x)
            if level < len(self.level_pools):
                x = self.level_pools[level](x)
        x = self.final_pool(x)
        x = F.relu(self.final_conv_0(x))
        x = F.relu(self.final_conv_1(x))
        return torch.mean(x, dim=(2, 3, 4))

    def load_from_tensorflow(self, tf_model_path: str, *, prefix: str = "qc") -> None:
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "h5py is required to load TensorFlow .h5 model weights"
            ) from e

        tf_weights: dict[str, np.ndarray] = {}
        with h5py.File(tf_model_path, "r") as f:

            def recursive_weight_loader(name, obj):
                if isinstance(obj, h5py.Dataset):
                    tf_weights[name] = np.array(obj)

            f.visititems(recursive_weight_loader)

        state_dict = OrderedDict(self.state_dict())
        for level in range(len(self.encoder_levels)):
            for conv_name, conv_idx in [("conv0", 0), ("conv1", 1)]:
                layer_prefix = (
                    f"{prefix}_conv_downarm_{level}_{conv_idx}/"
                    f"{prefix}_conv_downarm_{level}_{conv_idx}"
                )
                tf_name_conv = f"{layer_prefix}/kernel:0"
                tf_name_bias = f"{layer_prefix}/bias:0"
                weight = np.transpose(tf_weights[tf_name_conv], (4, 3, 0, 1, 2))
                state_dict[f"encoder_levels.{level}.{conv_name}.weight"] = (
                    torch.from_numpy(weight)
                )
                state_dict[f"encoder_levels.{level}.{conv_name}.bias"] = (
                    torch.from_numpy(tf_weights[tf_name_bias])
                )

            expand_prefix = (
                f"{prefix}_expand_down_merge_{level}/{prefix}_expand_down_merge_{level}"
            )
            expand_conv = f"{expand_prefix}/kernel:0"
            expand_bias = f"{expand_prefix}/bias:0"
            weight = np.transpose(tf_weights[expand_conv], (4, 3, 0, 1, 2))
            state_dict[f"encoder_levels.{level}.expand_conv.weight"] = torch.from_numpy(
                weight
            )
            state_dict[f"encoder_levels.{level}.expand_conv.bias"] = torch.from_numpy(
                tf_weights[expand_bias]
            )

            bn_prefix = f"{prefix}_bn_down_{level}/{prefix}_bn_down_{level}"
            state_dict[f"encoder_levels.{level}.batch_norm.weight"] = torch.from_numpy(
                tf_weights[f"{bn_prefix}/gamma:0"]
            )
            state_dict[f"encoder_levels.{level}.batch_norm.bias"] = torch.from_numpy(
                tf_weights[f"{bn_prefix}/beta:0"]
            )
            state_dict[f"encoder_levels.{level}.batch_norm.running_mean"] = (
                torch.from_numpy(tf_weights[f"{bn_prefix}/moving_mean:0"])
            )
            state_dict[f"encoder_levels.{level}.batch_norm.running_var"] = (
                torch.from_numpy(tf_weights[f"{bn_prefix}/moving_variance:0"])
            )

        for idx in [0, 1]:
            layer_prefix = f"{prefix}_final_conv_{idx}/{prefix}_final_conv_{idx}"
            tf_name_conv = f"{layer_prefix}/kernel:0"
            tf_name_bias = f"{layer_prefix}/bias:0"
            weight = np.transpose(tf_weights[tf_name_conv], (4, 3, 0, 1, 2))
            state_dict[f"final_conv_{idx}.weight"] = torch.from_numpy(weight)
            state_dict[f"final_conv_{idx}.bias"] = torch.from_numpy(
                tf_weights[tf_name_bias]
            )

        self.load_state_dict(state_dict)

    def load_from_pytorch(self, pt_model_path: str) -> None:
        checkpoint = torch.load(pt_model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        if isinstance(state_dict, dict) and "qc_model" in state_dict:
            state_dict = state_dict["qc_model"]
        self.load_state_dict(state_dict)

    def load_weights(self, model_path: str) -> None:
        model_path = str(model_path)
        if model_path.endswith(".h5"):
            self.load_from_tensorflow(model_path, prefix="qc")
            return
        if model_path.endswith(".pt"):
            self.load_from_pytorch(model_path)
            return
        raise ValueError(f"Unsupported model format: {model_path}")
