from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def forward(self, src: torch.Tensor, sample_coords_zyx: torch.Tensor) -> torch.Tensor:
        spatial_shape = (src.shape[-3], src.shape[-2], src.shape[-1])
        grid = voxel_to_normalized(sample_coords_zyx[..., [2, 1, 0]], spatial_shape)
        return F.grid_sample(
            src,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )


class VecInt(nn.Module):
    def __init__(self, *, int_steps: int = 10):
        super().__init__()
        self.int_steps = int_steps
        self.scale = 1.0 / (2**self.int_steps)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        if self.int_steps <= 0:
            return flow
        integrated = flow * self.scale
        for _ in range(self.int_steps):
            integrated = integrated + warp_flow(integrated, integrated)
        return integrated


class ResizeTransform(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        if self.factor == 1:
            return flow
        if self.factor < 1:
            flow = F.interpolate(
                flow,
                scale_factor=self.factor,
                mode="trilinear",
                align_corners=True,
                recompute_scale_factor=False,
            )
            return flow * self.factor
        flow = flow * self.factor
        return F.interpolate(
            flow,
            scale_factor=self.factor,
            mode="trilinear",
            align_corners=True,
            recompute_scale_factor=False,
        )


class EasyRegConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class EasyRegUnet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        encoder_features: list[int],
        decoder_features: list[int],
        half_res: bool,
    ):
        super().__init__()
        self.half_res = half_res
        self.encoder_features = encoder_features
        self.decoder_features = decoder_features
        self.nb_levels = len(encoder_features) + 1

        self.encoder = nn.ModuleList()
        prev_nf = in_channels
        encoder_nfs = [prev_nf]
        for out_channels in encoder_features:
            convs = nn.ModuleList([EasyRegConvBlock(prev_nf, out_channels)])
            self.encoder.append(convs)
            prev_nf = out_channels
            encoder_nfs.append(prev_nf)

        self.pooling = [nn.MaxPool3d(kernel_size=2) for _ in range(self.nb_levels)]
        self.upsampling = [nn.Upsample(scale_factor=2, mode="nearest") for _ in range(self.nb_levels)]

        reversed_encoder_nfs = list(np.flip(np.asarray(encoder_nfs)).tolist())
        decoder_level_features = decoder_features[: len(encoder_features)]
        self.decoder = nn.ModuleList()
        for level, out_channels in enumerate(decoder_level_features):
            convs = nn.ModuleList([EasyRegConvBlock(prev_nf, out_channels)])
            self.decoder.append(convs)
            prev_nf = out_channels
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += reversed_encoder_nfs[level]

        self.remaining = nn.ModuleList()
        for out_channels in decoder_features[len(encoder_features) :]:
            self.remaining.append(EasyRegConvBlock(prev_nf, out_channels))
            prev_nf = out_channels

        self.final_channels = prev_nf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_history: list[torch.Tensor] = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if (not self.half_res) or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        for block in self.remaining:
            x = block(x)

        return x


def voxel_to_normalized(coords_zyx: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    depth, height, width = spatial_shape
    z = coords_zyx[..., 0]
    y = coords_zyx[..., 1]
    x = coords_zyx[..., 2]
    x_norm = 2.0 * x / max(width - 1, 1) - 1.0
    y_norm = 2.0 * y / max(height - 1, 1) - 1.0
    z_norm = 2.0 * z / max(depth - 1, 1) - 1.0
    return torch.stack([x_norm, y_norm, z_norm], dim=-1)


def warp_flow(flow: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    device = flow.device
    batch, _, depth, height, width = flow.shape
    z, y, x = torch.meshgrid(
        torch.arange(depth, dtype=flow.dtype, device=device),
        torch.arange(height, dtype=flow.dtype, device=device),
        torch.arange(width, dtype=flow.dtype, device=device),
        indexing="ij",
    )
    base = torch.stack([z, y, x], dim=0).unsqueeze(0).expand(batch, -1, -1, -1, -1)
    sample_coords = base + displacement
    sample_coords = sample_coords.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
    grid = voxel_to_normalized(sample_coords, (depth, height, width))
    warped = F.grid_sample(
        flow,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped


class EasyRegDeformableNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 2,
        int_steps: int = 10,
        half_res: bool = True,
    ):
        super().__init__()
        self.unet = EasyRegUnet(
            in_channels=in_channels,
            encoder_features=[256, 256, 256, 256],
            decoder_features=[256, 256, 256, 256, 256, 256],
            half_res=half_res,
        )
        self.half_res = half_res
        self.flow = nn.Conv3d(self.unet.final_channels, 3, kernel_size=3, padding=1)
        self.resize = None
        self.fullsize = ResizeTransform(2.0) if int_steps > 0 and half_res else None
        self.integrate = VecInt(int_steps=int_steps)
        self.transformer = SpatialTransformer()
        nn.init.normal_(self.flow.weight, mean=0.0, std=1e-5)
        if self.flow.bias is not None:
            nn.init.zeros_(self.flow.bias)

    def forward(self, reference: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        features = self.unet(torch.cat([reference, moving], dim=1))
        flow = self.flow(features)
        flow = self.integrate(flow)
        if self.fullsize is not None:
            flow = self.fullsize(flow)
        return flow.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

    def load_from_tensorflow(self, tf_model_path: str, *, prefix: str = "vxm_dense") -> None:
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required to load TensorFlow .h5 model weights") from e

        tf_weights: dict[str, np.ndarray] = {}
        with h5py.File(tf_model_path, "r") as f:

            def recursive_weight_loader(name, obj):
                if isinstance(obj, h5py.Dataset):
                    tf_weights[name] = np.array(obj)

            f.visititems(recursive_weight_loader)

        state_dict = OrderedDict(self.state_dict())

        for level in range(len(self.unet.encoder_features)):
            layer_prefix = f"model_weights/{prefix}/{prefix}_unet_enc_conv_{level}_0"
            weight = np.transpose(tf_weights[f"{layer_prefix}/kernel:0"], (4, 3, 0, 1, 2))
            state_dict[f"unet.encoder.{level}.0.conv.weight"] = torch.from_numpy(weight)
            state_dict[f"unet.encoder.{level}.0.conv.bias"] = torch.from_numpy(tf_weights[f"{layer_prefix}/bias:0"])

        decoder_level_map = list(reversed(range(len(self.unet.encoder_features))))
        for level, tf_level in enumerate(decoder_level_map):
            layer_prefix = f"model_weights/{prefix}/{prefix}_unet_dec_conv_{tf_level}_0"
            weight = np.transpose(tf_weights[f"{layer_prefix}/kernel:0"], (4, 3, 0, 1, 2))
            state_dict[f"unet.decoder.{level}.0.conv.weight"] = torch.from_numpy(weight)
            state_dict[f"unet.decoder.{level}.0.conv.bias"] = torch.from_numpy(tf_weights[f"{layer_prefix}/bias:0"])

        layer_prefix = f"model_weights/{prefix}/{prefix}_unet_dec_final_conv_0"
        weight = np.transpose(tf_weights[f"{layer_prefix}/kernel:0"], (4, 3, 0, 1, 2))
        state_dict["unet.remaining.0.conv.weight"] = torch.from_numpy(weight)
        state_dict["unet.remaining.0.conv.bias"] = torch.from_numpy(tf_weights[f"{layer_prefix}/bias:0"])

        layer_prefix = f"model_weights/{prefix}/{prefix}_unet_dec_final_conv_1"
        weight = np.transpose(tf_weights[f"{layer_prefix}/kernel:0"], (4, 3, 0, 1, 2))
        state_dict["unet.remaining.1.conv.weight"] = torch.from_numpy(weight)
        state_dict["unet.remaining.1.conv.bias"] = torch.from_numpy(tf_weights[f"{layer_prefix}/bias:0"])

        flow_prefix = f"model_weights/{prefix}/{prefix}_flow"
        flow_weight = np.transpose(tf_weights[f"{flow_prefix}/kernel:0"], (4, 3, 0, 1, 2))
        state_dict["flow.weight"] = torch.from_numpy(flow_weight)
        state_dict["flow.bias"] = torch.from_numpy(tf_weights[f"{flow_prefix}/bias:0"])

        self.load_state_dict(state_dict)

    def load_from_pytorch(self, pt_model_path: str) -> None:
        checkpoint = torch.load(pt_model_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        )
        if isinstance(state_dict, dict) and "easyreg_model" in state_dict:
            state_dict = state_dict["easyreg_model"]
        self.load_state_dict(state_dict)

    def load_weights(self, model_path: str) -> None:
        model_path = str(model_path)
        if model_path.endswith(".h5"):
            self.load_from_tensorflow(model_path)
            return
        if model_path.endswith(".pt"):
            self.load_from_pytorch(model_path)
            return
        raise ValueError(f"Unsupported model format: {model_path}")
