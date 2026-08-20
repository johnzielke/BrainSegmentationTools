# ruff: noqa
# ty: noqa
from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import nibabel as nib
import nibabel.affines
import numpy as np
import torch


def _normalize_intensity(image: np.ndarray) -> np.ndarray:
    positive = image[image > 0]
    values = positive if positive.size else image.reshape(-1)
    min_value = float(np.percentile(values, 0.5))
    max_value = float(np.percentile(values, 99.5))
    image = np.clip(image, min_value, max_value)
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def _save_nifti(data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine, header=header.copy()), path.as_posix())


def _make_elastic_moving_image(reference_path: Path, moving_path: Path) -> tuple[Path, Path]:
    image = cast(nib.Nifti1Image, nib.load(reference_path.as_posix()))
    reference = np.asarray(image.get_fdata(dtype=np.float32))[:128, :128, :128]
    shape = reference.shape

    z, y, x = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    center = np.asarray(shape, dtype=np.float32) / 2.0
    dz = 1.75 * np.exp(
        -(
            ((z - center[0]) ** 2) / (2.0 * 16.0**2)
            + ((y - center[1]) ** 2) / (2.0 * 24.0**2)
            + ((x - center[2]) ** 2) / (2.0 * 18.0**2)
        )
    )
    dy = -1.25 * np.exp(
        -(
            ((z - (center[0] + 10.0)) ** 2) / (2.0 * 18.0**2)
            + ((y - (center[1] - 12.0)) ** 2) / (2.0 * 20.0**2)
            + ((x - center[2]) ** 2) / (2.0 * 22.0**2)
        )
    )
    dx = 1.0 * np.exp(
        -(
            ((z - (center[0] - 8.0)) ** 2) / (2.0 * 20.0**2)
            + ((y - center[1]) ** 2) / (2.0 * 18.0**2)
            + ((x - (center[2] + 14.0)) ** 2) / (2.0 * 16.0**2)
        )
    )
    sample_coords = np.stack([z + dz, y + dy, x + dx], axis=-1)
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

    _save_nifti(reference.astype(np.float32), image.affine, image.header, reference_path)
    _save_nifti(moving, image.affine, image.header, moving_path)
    return reference_path, moving_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EasyReg deformable-only oracle outputs")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--field-out", required=True)
    parser.add_argument("--warped-out", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    import tensorflow as tf
    import voxelmorph as vxm
    from keras import layers as KL

    reference_path = Path(args.reference)
    moving_path = Path(args.moving)
    field_out = Path(args.field_out)
    warped_out = Path(args.warped_out)

    reference_path, moving_path = _make_elastic_moving_image(reference_path, moving_path)
    reference_image = cast(nib.Nifti1Image, nib.load(reference_path.as_posix()))
    moving_image = cast(nib.Nifti1Image, nib.load(moving_path.as_posix()))
    reference = _normalize_intensity(np.asarray(reference_image.get_fdata(dtype=np.float32)))
    moving = _normalize_intensity(np.asarray(moving_image.get_fdata(dtype=np.float32)))

    if reference.shape != moving.shape:
        raise ValueError(f"reference and moving shapes must match, got {reference.shape} and {moving.shape}")

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
    cnn.load_weights(args.model_path, by_name=True)
    svf1 = cnn([source, target])[1]
    svf2 = cnn([target, source])[1]
    pos_svf = KL.Lambda(lambda x: 0.5 * x[0] - 0.5 * x[1])([svf1, svf2])
    neg_svf = KL.Lambda(lambda x: -x)(pos_svf)
    pos_def_small = vxm.layers.VecInt(method="ss", int_steps=10)(pos_svf)
    neg_def_small = vxm.layers.VecInt(method="ss", int_steps=10)(neg_svf)
    pos_def = vxm.layers.RescaleTransform(2)(pos_def_small)
    neg_def = vxm.layers.RescaleTransform(2)(neg_def_small)
    model = tf.keras.Model(inputs=[source, target], outputs=[pos_def, neg_def])
    model.load_weights(args.model_path)

    pred = model.predict(
        [reference[np.newaxis, ..., np.newaxis], moving[np.newaxis, ..., np.newaxis]],
        verbose=0,
    )
    field_vox = np.asarray(pred[0][0], dtype=np.float32)

    z, y, x = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        np.arange(reference.shape[2], dtype=np.float32),
        indexing="ij",
    )
    total_grid = np.stack([z, y, x], axis=-1) + field_vox
    ras_coords = nibabel.affines.apply_affine(moving_image.affine, total_grid[..., ::-1]).astype(np.float32)
    _save_nifti(ras_coords, reference_image.affine, reference_image.header, field_out)

    from brain_segmentation_tools.easyreg.app import _warp_image

    moving_tensor = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0)
    warped = _warp_image(moving_tensor, torch.from_numpy(total_grid)).squeeze().numpy().astype(np.float32)
    _save_nifti(warped, reference_image.affine, reference_image.header, warped_out)


if __name__ == "__main__":
    main()