# ruff: noqa
# ty: noqa
from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import torch

from brain_segmentation_tools.easyreg.model import EasyRegDeformableNet
from brain_segmentation_tools.model_manager import ModelManager


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


def main() -> None:
    reference, moving = _make_test_pair()

    import tensorflow as tf
    import voxelmorph as vxm
    from keras import layers as KL

    source = tf.keras.Input(shape=(*reference.shape, 1))
    target = tf.keras.Input(shape=(*reference.shape, 1))
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
    model_path = Path("dev/freesurfer/mri_easyreg/easyreg_v10_230103.h5")
    cnn.load_weights(model_path.as_posix(), by_name=True)
    svf1 = cnn([source, target])[1]
    svf2 = cnn([target, source])[1]
    pos_svf = KL.Lambda(lambda x: 0.5 * x[0] - 0.5 * x[1])([svf1, svf2])
    pos_def_small = vxm.layers.VecInt(method="ss", int_steps=10)(pos_svf)
    pos_def = vxm.layers.RescaleTransform(2)(pos_def_small)
    tf_model = tf.keras.Model(inputs=[source, target], outputs=pos_def)
    tf_model.load_weights(model_path.as_posix())
    tf_output = np.asarray(
        tf_model.predict([reference[np.newaxis, ..., np.newaxis], moving[np.newaxis, ..., np.newaxis]], verbose=0),
        dtype=np.float32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = Path(tmpdir) / "easyreg.pt"
        ModelManager(dev_mode=True).convert_h5_to_pt(
            model_name="easyreg",
            model_type="deformable_field",
            version="1.0",
            output_path=pt_path,
        )
        pt_model = EasyRegDeformableNet()
        pt_model.load_weights(pt_path.as_posix())
        pt_model.eval()
        with torch.inference_mode():
            pt_output = pt_model(
                torch.from_numpy(reference).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(moving).unsqueeze(0).unsqueeze(0),
            ).cpu().numpy()

    print(f"tf_output_shape={tf_output.shape}")
    print(f"pt_output_shape={pt_output.shape}")
    try:
        diff = np.abs(tf_output - pt_output)
        print(f"max_abs_diff={float(diff.max()):.8f}")
        print(f"mean_abs_diff={float(diff.mean()):.8f}")
    except ValueError as e:
        print(f"ValueError during subtraction: {e}")

if __name__ == '__main__':
    main()
