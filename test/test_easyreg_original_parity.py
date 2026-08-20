from __future__ import annotations

from pathlib import Path
from typing import cast

import nibabel as nib
import nibabel.affines
import numpy as np
import pytest
import torch

from brain_segmentation_tools.easyreg.app import EasyRegApplication
from brain_segmentation_tools.model_manager import ModelManager
from test.helpers import ORACLE_DIR, TEST_RES_DIR, load_nifti

FIELD_MAX_ABS_DIFF = 5e-4
FIELD_MEAN_ABS_DIFF = 1e-5
WARPED_MAX_ABS_DIFF = 5e-4
WARPED_MEAN_ABS_DIFF = 1e-5
EASYREG_INPUT_SHAPE = (128, 128, 128)


def _absolute_ras_field_to_voxel_displacement(field_ras: np.ndarray, moving_affine: np.ndarray) -> np.ndarray:
    ijk = nib.affines.apply_affine(np.linalg.inv(moving_affine), field_ras)
    z, y, x = np.meshgrid(
        np.arange(field_ras.shape[0], dtype=np.float32),
        np.arange(field_ras.shape[1], dtype=np.float32),
        np.arange(field_ras.shape[2], dtype=np.float32),
        indexing="ij",
    )
    base_xyz = np.stack([x, y, z], axis=-1)
    return (ijk - base_xyz).astype(np.float32)


def _make_deformed_input(reference_path: Path, output_path: Path) -> Path:
    image = cast(nib.Nifti1Image, nib.load(reference_path.as_posix()))
    reference_full = np.asarray(image.get_fdata(dtype=np.float32))
    reference = reference_full[: EASYREG_INPUT_SHAPE[0], : EASYREG_INPUT_SHAPE[1], : EASYREG_INPUT_SHAPE[2]]

    z, y, x = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        np.arange(reference.shape[2], dtype=np.float32),
        indexing="ij",
    )
    center = np.asarray(reference.shape, dtype=np.float32) / 2.0
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
        reference.shape,
        strict=True,
    ):
        np.clip(axis_coords, 0.0, float(axis_size - 1), out=axis_coords)

    z0 = np.floor(sample_coords[..., 0]).astype(np.int32)
    y0 = np.floor(sample_coords[..., 1]).astype(np.int32)
    x0 = np.floor(sample_coords[..., 2]).astype(np.int32)
    z1 = np.clip(z0 + 1, 0, reference.shape[0] - 1)
    y1 = np.clip(y0 + 1, 0, reference.shape[1] - 1)
    x1 = np.clip(x0 + 1, 0, reference.shape[2] - 1)

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
    deformed = (c0 * (1.0 - dz) + c1 * dz).astype(np.float32)

    affine = image.affine.copy()
    header = image.header.copy()
    nib.save(nib.Nifti1Image(deformed, affine, header=header), output_path)
    return output_path


def test_easyreg_matches_original_oracle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_cache_dir = tmp_path / "model_cache"
    monkeypatch.setenv(ModelManager.MODEL_CACHE_DIR_ENV_VAR, model_cache_dir.as_posix())
    manager = ModelManager(dev_mode=True)
    manager.convert_h5_to_pt(
        model_name="easyreg",
        model_type="deformable_field",
        version="1.0",
        output_path=model_cache_dir / "easyreg_deformable_field_1.0.pt",
    )

    source_reference = cast(nib.Nifti1Image, nib.load((TEST_RES_DIR / "spgr_unstrip.nii.gz").as_posix()))
    reference_path = tmp_path / "spgr_unstrip_easyreg_reference.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.asarray(source_reference.get_fdata(dtype=np.float32))[
                : EASYREG_INPUT_SHAPE[0], : EASYREG_INPUT_SHAPE[1], : EASYREG_INPUT_SHAPE[2]
            ],
            source_reference.affine,
            header=source_reference.header.copy(),
        ),
        reference_path,
    )
    moving_path = _make_deformed_input(reference_path, tmp_path / "spgr_unstrip_easyreg_moving.nii.gz")

    field_output = tmp_path / "easyreg_field.nii.gz"
    warped_output = tmp_path / "easyreg_warped.nii.gz"
    app = EasyRegApplication(device="cpu", dev_mode=False)
    torch.set_num_threads(1)
    app.predict_deformable_fields(
        reference_images=reference_path,
        moving_images=moving_path,
        deformation_field_out=field_output,
        warped_image_out=warped_output,
    )

    expected_field_path = ORACLE_DIR / "easyreg_oracle_deformation_field.nii.gz"
    expected_warped_path = ORACLE_DIR / "easyreg_oracle_warped_image.nii.gz"
    if not expected_field_path.exists() or not expected_warped_path.exists():
        pytest.skip("EasyReg oracle artifacts are missing; run dev/generate_easyreg_oracle.bash")

    actual_field, actual_field_affine = load_nifti(field_output)
    expected_field, expected_field_affine = load_nifti(expected_field_path)
    actual_warped, actual_warped_affine = load_nifti(warped_output)
    expected_warped, expected_warped_affine = load_nifti(expected_warped_path)

    assert actual_field.shape == expected_field.shape
    assert actual_warped.shape == expected_warped.shape
    assert np.allclose(actual_field_affine, expected_field_affine, atol=1e-5)
    assert np.allclose(actual_warped_affine, expected_warped_affine, atol=1e-5)

    moving_affine = cast(nib.Nifti1Image, nib.load(moving_path.as_posix())).affine
    actual_field_disp = _absolute_ras_field_to_voxel_displacement(actual_field.astype(np.float32), moving_affine)
    expected_field_disp = _absolute_ras_field_to_voxel_displacement(expected_field.astype(np.float32), moving_affine)
    field_diff = np.abs(actual_field_disp - expected_field_disp)
    warped_diff = np.abs(actual_warped.astype(np.float32) - expected_warped.astype(np.float32))
    field_max_abs_diff = float(field_diff.max())
    field_mean_abs_diff = float(field_diff.mean())
    warped_max_abs_diff = float(warped_diff.max())
    warped_mean_abs_diff = float(warped_diff.mean())
    print(
        "EasyReg oracle metrics: "
        f"field_max_abs_diff={field_max_abs_diff:.8f}, "
        f"field_mean_abs_diff={field_mean_abs_diff:.8f}, "
        f"warped_max_abs_diff={warped_max_abs_diff:.8f}, "
        f"warped_mean_abs_diff={warped_mean_abs_diff:.8f}"
    )

    assert field_mean_abs_diff <= FIELD_MEAN_ABS_DIFF
    assert field_max_abs_diff <= FIELD_MAX_ABS_DIFF
    assert warped_mean_abs_diff <= WARPED_MEAN_ABS_DIFF
    assert warped_max_abs_diff <= WARPED_MAX_ABS_DIFF
