from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from brain_segmentation_tools.easyreg.model import EasyRegDeformableNet


if not hasattr(inspect, "getargspec"):
    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return spec.args, spec.varargs, spec.varkw, spec.defaults

    inspect.getargspec = _getargspec


def _normalize_intensity(image: np.ndarray) -> np.ndarray:
    positive = image[image > 0]
    values = positive if positive.size else image.reshape(-1)
    min_value = float(np.percentile(values, 0.5))
    max_value = float(np.percentile(values, 99.5))
    image = np.clip(image, min_value, max_value)
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def _make_test_pair(shape: tuple[int, int, int] = (128, 128, 128)) -> tuple[np.ndarray, np.ndarray]:
    z, y, x = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    center = np.asarray(shape, dtype=np.float32) / 2.0
    reference = np.exp(
        -(
            ((z - center[0]) ** 2) / (2.0 * 28.0**2)
            + ((y - center[1]) ** 2) / (2.0 * 24.0**2)
            + ((x - center[2]) ** 2) / (2.0 * 20.0**2)
        )
    ).astype(np.float32)
    shift_z = 1.75 * np.exp(
        -(
            ((z - center[0]) ** 2) / (2.0 * 16.0**2)
            + ((y - center[1]) ** 2) / (2.0 * 24.0**2)
            + ((x - center[2]) ** 2) / (2.0 * 18.0**2)
        )
    )
    shift_y = -1.25 * np.exp(
        -(
            ((z - (center[0] + 10.0)) ** 2) / (2.0 * 18.0**2)
            + ((y - (center[1] - 12.0)) ** 2) / (2.0 * 20.0**2)
            + ((x - center[2]) ** 2) / (2.0 * 22.0**2)
        )
    )
    shift_x = 1.0 * np.exp(
        -(
            ((z - (center[0] - 8.0)) ** 2) / (2.0 * 20.0**2)
            + ((y - center[1]) ** 2) / (2.0 * 18.0**2)
            + ((x - (center[2] + 14.0)) ** 2) / (2.0 * 16.0**2)
        )
    )
    sample_coords = np.stack([z + shift_z, y + shift_y, x + shift_x], axis=-1)
    for axis_coords, axis_size in zip(
        (sample_coords[..., 0], sample_coords[..., 1], sample_coords[..., 2]),
        shape,
        strict=True,
    ):
        np.clip(axis_coords, 0.0, float(axis_size - 1), out=axis_coords)

    z0 = np.floor(sample_coords[..., 0]).astype(np.int32)
    y0 = np.floor(sample_coords[..., 1]).astype(np.int32)
    x0 = np.floor(sample_coords[..., 2]).astype(np.int32)
    z1 = np.clip(z0 + 1, 0, shape[0] - 1)
    y1 = np.clip(y0 + 1, 0, shape[1] - 1)
    x1 = np.clip(x0 + 1, 0, shape[2] - 1)
    dz = sample_coords[..., 0] - z0
    dy = sample_coords[..., 1] - y0
    dx = sample_coords[..., 2] - x0

    c000 = reference[z0, y0, x0]
    c001 = reference[z0, y0, x1]
    c010 = reference[z0, y1, x0]
    c011 = reference[z0, y1, x1]
    c100 = reference[z1, y0, x0]
    c101 = reference[z1, y0, x1]
    c110 = reference[z1, y1, x0]
    c111 = reference[z1, y1, x1]

    c00 = c000 * (1.0 - dx) + c001 * dx
    c01 = c010 * (1.0 - dx) + c011 * dx
    c10 = c100 * (1.0 - dx) + c101 * dx
    c11 = c110 * (1.0 - dx) + c111 * dx
    c0 = c00 * (1.0 - dy) + c01 * dy
    c1 = c10 * (1.0 - dy) + c11 * dy
    moving = (c0 * (1.0 - dz) + c1 * dz).astype(np.float32)
    return _normalize_intensity(reference), _normalize_intensity(moving)


def _print_diff(name: str, tf_tensor: np.ndarray, pt_tensor: np.ndarray) -> None:
    diff = np.abs(tf_tensor.astype(np.float32) - pt_tensor.astype(np.float32))
    print(
        f"{name}: shape_tf={tuple(tf_tensor.shape)} shape_pt={tuple(pt_tensor.shape)} "
        f"max_abs_diff={float(diff.max()):.8f} mean_abs_diff={float(diff.mean()):.8f}"
    )


def main() -> None:
    try:
        import keras.src.losses.losses as losses
        import tensorflow.keras.losses as tf_losses

        tf_losses.mean_absolute_error = losses.mean_absolute_error
        tf_losses.mean_squared_error = losses.mean_squared_error
    except ImportError:
        pass

    import tensorflow as tf
    import voxelmorph as vxm

    reference, moving = _make_test_pair()
    model_path = Path("dev/freesurfer/mri_easyreg/easyreg_v10_230103.h5")
    config = {
        "name": "vxm_dense",
        "fill_value": None,
        "input_model": None,
        "unet_half_res": True,
        "trg_feats": 1,
        "src_feats": 1,
        "use_probs": False,
        "bidir": False,
        "int_downsize": 2,
        "int_steps": 10,
        "nb_unet_conv_per_level": 1,
        "unet_feat_mult": 1,
        "nb_unet_levels": None,
        "nb_unet_features": [[256, 256, 256, 256], [256, 256, 256, 256, 256, 256]],
        "inshape": list(reference.shape),
    }
    cnn = vxm.networks.VxmDense(**config)
    cnn.load_weights(model_path.as_posix(), by_name=True)

    layer_names = [
        "vxm_dense_unet_enc_conv_0_0",
        "vxm_dense_unet_enc_conv_1_0",
        "vxm_dense_unet_enc_conv_2_0",
        "vxm_dense_unet_enc_conv_3_0",
        "vxm_dense_unet_dec_conv_3_0",
        "vxm_dense_unet_dec_conv_2_0",
        "vxm_dense_unet_dec_conv_1_0",
        "vxm_dense_unet_dec_conv_0_0",
        "vxm_dense_unet_dec_final_conv_0",
        "vxm_dense_unet_dec_final_conv_1",
        "vxm_dense_flow",
    ]
    tf_probe = tf.keras.Model(inputs=cnn.inputs, outputs=[cnn.get_layer(name).output for name in layer_names])
    tf_outputs = tf_probe.predict(
        [reference[np.newaxis, ..., np.newaxis], moving[np.newaxis, ..., np.newaxis]],
        verbose=0,
    )

    pt_model = EasyRegDeformableNet()
    pt_model.load_from_tensorflow(model_path.as_posix())
    pt_model.eval()
    pt_intermediates: dict[str, np.ndarray] = {}

    def _capture(name: str):
        def _hook(_module, _inputs, output):
            pt_intermediates[name] = output.detach().cpu().numpy()

        return _hook

    hooks = []
    hooks.extend(
        block[0].register_forward_hook(_capture(f"enc_{index}"))
        for index, block in enumerate(pt_model.unet.encoder)
    )
    hooks.extend(
        block[0].register_forward_hook(_capture(f"dec_{index}"))
        for index, block in enumerate(pt_model.unet.decoder)
    )
    hooks.extend(
        block.register_forward_hook(_capture(f"remaining_{index}"))
        for index, block in enumerate(pt_model.unet.remaining)
    )
    hooks.append(pt_model.flow.register_forward_hook(_capture("flow")))

    with torch.inference_mode():
        _ = pt_model(
            torch.from_numpy(reference).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(moving).unsqueeze(0).unsqueeze(0),
        )

    for hook in hooks:
        hook.remove()

    tf_named = {
        "enc_0": tf_outputs[0].transpose(0, 4, 1, 2, 3),
        "enc_1": tf_outputs[1].transpose(0, 4, 1, 2, 3),
        "enc_2": tf_outputs[2].transpose(0, 4, 1, 2, 3),
        "enc_3": tf_outputs[3].transpose(0, 4, 1, 2, 3),
        "dec_0": tf_outputs[4].transpose(0, 4, 1, 2, 3),
        "dec_1": tf_outputs[5].transpose(0, 4, 1, 2, 3),
        "dec_2": tf_outputs[6].transpose(0, 4, 1, 2, 3),
        "dec_3": tf_outputs[7].transpose(0, 4, 1, 2, 3),
        "remaining_0": tf_outputs[8].transpose(0, 4, 1, 2, 3),
        "remaining_1": tf_outputs[9].transpose(0, 4, 1, 2, 3),
        "flow": tf_outputs[10].transpose(0, 4, 1, 2, 3),
    }

    for name in [
        "enc_0",
        "enc_1",
        "enc_2",
        "enc_3",
        "dec_0",
        "dec_1",
        "dec_2",
        "dec_3",
        "remaining_0",
        "remaining_1",
        "flow",
    ]:
        _print_diff(name, tf_named[name], pt_intermediates[name])


if __name__ == "__main__":
    main()
