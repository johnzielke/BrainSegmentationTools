from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_unknown_state_dict_keys(model: nn.Module, state_dict: dict) -> dict:
    """Drop entries for parameters the model no longer defines (e.g. stale upsample weights)."""
    own_keys = set(model.state_dict().keys())
    return {key: value for key, value in state_dict.items() if key in own_keys}


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding="same",
        activation="elu",
        use_residual=False,
        conv_dropout=0,
        batch_norm=None,
        is_last_in_level=False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.activation = activation

        if padding == "same":
            padding = kernel_size // 2

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm is not None and is_last_in_level else None
        self.dropout = nn.Dropout3d(p=conv_dropout) if conv_dropout > 0 else None

    def forward(self, x):
        identity = x

        out = self.conv(x)

        if self.dropout is not None:
            out = self.dropout(out)

        if self.use_residual:
            if identity.size(1) != out.size(1):
                identity = F.pad(identity, (0, 0, 0, 0, 0, 0, 0, out.size(1) - identity.size(1)))
            out = out + identity

        if self.activation == "elu":
            out = F.elu(out)
        elif self.activation == "leaky_relu":
            out = F.leaky_relu(out, negative_slope=0.2)
        elif self.activation == "relu":
            out = F.relu(out)
        if self.batch_norm is not None:
            before_norm = out
            out = self.batch_norm(out)

            return out, before_norm
        else:
            return out


def upsample_layer(channels, factor=2):
    return nn.Upsample(scale_factor=factor, mode="nearest")


class UNet(nn.Module):
    def __init__(
        self,
        nb_features,
        in_channels,
        nb_levels,
        conv_size,
        nb_labels,
        feat_mult=1,
        pool_size=2,
        padding="same",
        activation="elu",
        use_residuals=False,
        final_pred_activation="softmax",
        nb_conv_per_level=1,
        skip_n_concatenations=0,
        half_res=False,
        conv_dropout=0,
        batch_norm=None,
    ):
        super().__init__()

        self.nb_levels = nb_levels
        self.final_pred_activation = final_pred_activation
        self.padding = padding
        self.pool_size = pool_size
        self.nb_conv_per_level = nb_conv_per_level
        self.skip_n_concatenations = skip_n_concatenations
        self.half_res = half_res

        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for level in range(nb_levels):
            level_features = []
            nb_lvl_feats = int(nb_features * feat_mult**level)

            for conv in range(nb_conv_per_level):
                block = ConvBlock(
                    in_channels,
                    nb_lvl_feats,
                    conv_size,
                    padding=padding,
                    activation=activation,
                    use_residual=use_residuals and conv == nb_conv_per_level - 1,
                    conv_dropout=conv_dropout,
                    batch_norm=batch_norm,
                    is_last_in_level=conv == nb_conv_per_level - 1,
                )
                level_features.append(block)
                in_channels = nb_lvl_feats

            self.encoder_blocks.append(nn.Sequential(*level_features))

            if level < nb_levels - 1:
                self.pool_layers.append(nn.MaxPool3d(pool_size))

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoder_uses_skip_connections: list[bool] = []

        for decoder_level, level in enumerate(range(nb_levels - 2, -1, -1)):
            nb_lvl_feats = int(nb_features * feat_mult**level)
            use_skip_connections = decoder_level < (nb_levels - self.skip_n_concatenations - 1)
            self.decoder_uses_skip_connections.append(use_skip_connections)
            should_upsample = (not self.half_res) or decoder_level < (nb_levels - 2)

            in_channels = int(nb_features * feat_mult ** (level + 1))
            if should_upsample and use_skip_connections:
                in_channels += nb_lvl_feats
            self.upsamples.append(
                upsample_layer(
                    channels=int(nb_features * feat_mult ** (level + 1)),
                    factor=self.pool_size,
                )
            )

            level_features = []
            for conv in range(nb_conv_per_level):
                block = ConvBlock(
                    in_channels,
                    nb_lvl_feats,
                    conv_size,
                    padding=padding,
                    activation=activation,
                    use_residual=use_residuals and conv == nb_conv_per_level - 1,
                    conv_dropout=conv_dropout,
                    batch_norm=batch_norm,
                    is_last_in_level=conv == nb_conv_per_level - 1,
                )
                level_features.append(block)
                in_channels = nb_lvl_feats

            self.decoder_blocks.append(nn.Sequential(*level_features))

        self.final_conv = nn.Conv3d(nb_lvl_feats, nb_labels, 1)

    def forward(self, x):
        encoder_features = []

        for level in range(self.nb_levels):
            r_val = self.encoder_blocks[level](x)
            if isinstance(r_val, tuple):
                x, before_norm = r_val
            else:
                x, before_norm = r_val, r_val
            if level < self.nb_levels - 1:
                encoder_features.append(before_norm)
                x = self.pool_layers[level](x)

        for level in range(self.nb_levels - 1):
            should_upsample = (not self.half_res) or level < (self.nb_levels - 2)
            if should_upsample:
                x = self.upsamples[level](x)
            if should_upsample and self.decoder_uses_skip_connections[level]:
                x = torch.cat([encoder_features[-level - 1], x], dim=1)
            r_val = self.decoder_blocks[level](x)
            if isinstance(r_val, tuple):
                x, _ = r_val
            else:
                x = r_val

        x = self.final_conv(x)

        if self.final_pred_activation == "softmax":
            x = F.softmax(x, dim=1)

        return x

    def load_from_tensorflow(self, tf_model_path, prefix):
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required to load TensorFlow .h5 model weights") from e

        tf_weights = {}

        with h5py.File(tf_model_path, "r") as f:

            def recursive_weight_loader(name, obj):
                if isinstance(obj, h5py.Dataset):
                    tf_weights[name] = np.array(obj)

            f.visititems(recursive_weight_loader)

        state_dict = OrderedDict()

        # Transfer non-existing upsample weights
        for key, value in self.state_dict().items():
            if key.startswith("upsamples"):
                state_dict[key] = value

        for level in range(self.nb_levels):
            for conv_idx in range(self.nb_conv_per_level):
                layer_prefix = f"{prefix}_conv_downarm_{level}_{conv_idx}/{prefix}_conv_downarm_{level}_{conv_idx}"
                tf_name_conv = f"{layer_prefix}/kernel:0"
                tf_name_bias = f"{layer_prefix}/bias:0"

                if tf_name_conv in tf_weights:
                    weight = np.transpose(tf_weights[tf_name_conv], (4, 3, 0, 1, 2))
                    state_dict[f"encoder_blocks.{level}.{conv_idx}.conv.weight"] = torch.from_numpy(weight)
                    state_dict[f"encoder_blocks.{level}.{conv_idx}.conv.bias"] = torch.from_numpy(
                        tf_weights[tf_name_bias]
                    )

                if conv_idx == self.nb_conv_per_level - 1:
                    bn_prefix = f"{prefix}_bn_down_{level}/{prefix}_bn_down_{level}"
                    if f"{bn_prefix}/gamma:0" in tf_weights:
                        state_dict[f"encoder_blocks.{level}.{conv_idx}.batch_norm.weight"] = torch.from_numpy(
                            tf_weights[f"{bn_prefix}/gamma:0"]
                        )
                        state_dict[f"encoder_blocks.{level}.{conv_idx}.batch_norm.bias"] = torch.from_numpy(
                            tf_weights[f"{bn_prefix}/beta:0"]
                        )
                        state_dict[f"encoder_blocks.{level}.{conv_idx}.batch_norm.running_mean"] = torch.from_numpy(
                            tf_weights[f"{bn_prefix}/moving_mean:0"]
                        )
                        state_dict[f"encoder_blocks.{level}.{conv_idx}.batch_norm.running_var"] = torch.from_numpy(
                            tf_weights[f"{bn_prefix}/moving_variance:0"]
                        )

        for level in range(self.nb_levels - 1):
            for conv_idx in range(self.nb_conv_per_level):
                decoder_level = self.nb_levels + level
                layer_prefix = (
                    f"{prefix}_conv_uparm_{decoder_level}_{conv_idx}/{prefix}_conv_uparm_{decoder_level}_{conv_idx}"
                )
                tf_name_conv = f"{layer_prefix}/kernel:0"
                tf_name_bias = f"{layer_prefix}/bias:0"

                if tf_name_conv in tf_weights:
                    weight = np.transpose(tf_weights[tf_name_conv], (4, 3, 0, 1, 2))
                    state_dict[f"decoder_blocks.{level}.{conv_idx}.conv.weight"] = torch.from_numpy(weight)
                    state_dict[f"decoder_blocks.{level}.{conv_idx}.conv.bias"] = torch.from_numpy(
                        tf_weights[tf_name_bias]
                    )

                if conv_idx == self.nb_conv_per_level - 1:
                    bn_prefix = f"{prefix}_bn_up_{level}/{prefix}_bn_up_{level}"
                    if f"{bn_prefix}/gamma:0" in tf_weights:
                        state_dict[f"decoder_blocks.{level}.{conv_idx}.batch_norm.weight"] = torch.from_numpy(
                            tf_weights[f"{bn_prefix}/gamma:0"]
                        )
                        state_dict[f"decoder_blocks.{level}.{conv_idx}.batch_norm.bias"] = torch.from_numpy(
                            tf_weights[f"{bn_prefix}/beta:0"]
                        )
                        state_dict[f"decoder_blocks.{level}.{conv_idx}.batch_norm.running_mean"] = torch.from_numpy(
                            tf_weights[f"{bn_prefix}/moving_mean:0"]
                        )
                        state_dict[f"decoder_blocks.{level}.{conv_idx}.batch_norm.running_var"] = torch.from_numpy(
                            tf_weights[f"{bn_prefix}/moving_variance:0"]
                        )

        likelihood_prefix = f"{prefix}_likelihood/{prefix}_likelihood"
        tf_name_conv = f"{likelihood_prefix}/kernel:0"
        tf_name_bias = f"{likelihood_prefix}/bias:0"
        if tf_name_conv in tf_weights:
            weight = np.transpose(tf_weights[tf_name_conv], (4, 3, 0, 1, 2))
            state_dict["final_conv.weight"] = torch.from_numpy(weight)
            state_dict["final_conv.bias"] = torch.from_numpy(tf_weights[tf_name_bias])

        self.load_state_dict(state_dict)

    def load_from_pytorch(self, pt_model_path):
        checkpoint = torch.load(pt_model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.load_state_dict(drop_unknown_state_dict_keys(self, state_dict))

    def load_weights(self, model_path, *, prefix=None):
        model_path = str(model_path)
        if model_path.endswith(".h5"):
            if prefix is None:
                raise ValueError("prefix is required when loading .h5 TensorFlow weights")
            self.load_from_tensorflow(model_path, prefix=prefix)
            return
        if model_path.endswith(".pt"):
            self.load_from_pytorch(model_path)
            return
        raise ValueError(f"Unsupported model format: {model_path}")
