from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest

from brain_segmentation_tools.model_manager import ModelManager
from test.helpers import (
    ORACLE_DIR,
    TEST_RES_DIR,
    dice,
    install_test_models,
    load_nifti,
    patch_single_worker_dataloader,
    preferred_test_device,
)

SEGMENTATION_MIN_VOXEL_ACCURACY = 0.99
SEGMENTATION_MIN_FOREGROUND_DICE = 0.99


def _crop_nifti_to_brain_border(*, image_path: Path, brain_mask_path: Path, output_path: Path) -> tuple[int, Path]:
    image = cast(nib.Nifti1Image, nib.load(image_path.as_posix()))
    image_data = np.asanyarray(image.dataobj)
    brain_mask = cast(nib.Nifti1Image, nib.load(brain_mask_path.as_posix()))
    brain_mask_data = np.asanyarray(brain_mask.dataobj) > 0

    crop_start_x = int(np.argwhere(brain_mask_data)[:, 0].min())
    cropped_image = image_data[crop_start_x:, :, :]

    cropped_affine = image.affine.copy()
    cropped_affine[:3, 3] = image.affine[:3, :3] @ np.array([crop_start_x, 0, 0]) + image.affine[:3, 3]
    nib.save(
        nib.Nifti1Image(cropped_image.astype(np.float32), cropped_affine),
        output_path,
    )
    return crop_start_x, output_path


# @pytest.mark.xfail(
#     strict=True,
#     reason=(
#         "When the predicted brain mask reaches the image border, the margin added "
#         "to the SynthSeg crop can push the bounding box outside the tensor."
#     ),
# )
def test_run_with_cropped_input_and_brain_touching_border(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("monai")

    from brain_segmentation_tools.app import Application

    patch_single_worker_dataloader(monkeypatch)

    model_cache_dir = tmp_path / "model_cache"
    monkeypatch.setenv(ModelManager.MODEL_CACHE_DIR_ENV_VAR, model_cache_dir.as_posix())
    install_test_models(
        model_cache_dir,
        [
            ("synthseg", "segmentation", "2.0"),
            ("synthstrip", "normal", "1"),
        ],
    )

    cropped_input_path = tmp_path / "spgr_unstrip_brain_at_border.nii.gz"
    crop_start_x, _ = _crop_nifti_to_brain_border(
        image_path=TEST_RES_DIR / "spgr_unstrip.nii.gz",
        brain_mask_path=ORACLE_DIR / "synthstrip_oracle_nocsf_0.nii.gz",
        output_path=cropped_input_path,
    )

    output_path = tmp_path / "cropped_synthseg.nii.gz"
    brain_mask_output_path = tmp_path / "cropped_synthstrip.nii.gz"
    app = Application(
        device=preferred_test_device(),
        version="v2.0",
        no_compile=True,
        dev_mode=False,
        crop_segmentation_input_to_brain_mask=True,
    )
    app.run(
        input_paths=cropped_input_path.as_posix(),
        segmentation_out=output_path.as_posix(),
        brain_mask_out=brain_mask_output_path.as_posix(),
        use_prog_bar=False,
    )

    assert output_path.exists()
    assert brain_mask_output_path.exists()

    synthseg_oracle_path = ORACLE_DIR / "synthseg_oracle_robust_0_parc_0.nii.gz"
    synthstrip_oracle_path = ORACLE_DIR / "synthstrip_oracle_nocsf_0.nii.gz"
    assert synthseg_oracle_path.exists()
    assert synthstrip_oracle_path.exists()

    actual_data, actual_affine = load_nifti(output_path)
    expected_data, _expected_affine = load_nifti(synthseg_oracle_path)
    expected_data = expected_data[crop_start_x:, :, :]

    shutil.copy2(output_path, Path("debug/cropped_synthseg_for_debug.nii.gz"))
    shutil.copy2(brain_mask_output_path, Path("debug/cropped_synthstrip_for_debug.nii.gz"))
    shutil.copy2(cropped_input_path, Path("debug/cropped_input_for_debug.nii.gz"))
    _, _ = _crop_nifti_to_brain_border(
        image_path=synthseg_oracle_path,
        brain_mask_path=synthstrip_oracle_path,
        output_path=Path("debug/cropped_synthseg.nii.gz"),
    )
    # Write a diagnostic difference image for border-crop failures.
    debug_diff_path = Path("debug/cropped_synthseg_diff.nii.gz")
    diff_data = np.abs(actual_data.astype(np.float32) - expected_data.astype(np.float32))
    nib.save(
        nib.Nifti1Image(diff_data, actual_affine),
        debug_diff_path,
    )
    original_affine = cast(nib.Nifti1Image, nib.load(synthseg_oracle_path.as_posix())).affine
    expected_affine = original_affine.copy()
    expected_affine[:3, 3] = original_affine[:3, :3] @ np.array([crop_start_x, 0, 0]) + original_affine[:3, 3]

    assert actual_data.shape == expected_data.shape
    assert np.allclose(actual_affine, expected_affine, atol=1e-5)

    actual_labels = actual_data.astype(np.int32)
    expected_labels = expected_data.astype(np.int32)
    voxel_accuracy = float((actual_labels == expected_labels).mean())
    foreground_dice = dice(actual_labels, expected_labels)
    print(
        f"Cropped-border SynthSeg metrics: voxel_accuracy={voxel_accuracy:.6f}, foreground_dice={foreground_dice:.6f}"
    )

    assert voxel_accuracy >= SEGMENTATION_MIN_VOXEL_ACCURACY
    assert foreground_dice >= SEGMENTATION_MIN_FOREGROUND_DICE


def test_run_with_contrast_classifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("monai")

    from brain_segmentation_tools.app import Application

    patch_single_worker_dataloader(monkeypatch)

    model_cache_dir = tmp_path / "model_cache"
    monkeypatch.setenv(ModelManager.MODEL_CACHE_DIR_ENV_VAR, model_cache_dir.as_posix())
    install_test_models(
        model_cache_dir,
        [
            ("synthseg", "segmentation", "2.0"),
            ("synthstrip", "normal", "1"),
            ("contrast_classifier", "normal", "1"),
        ],
    )

    app = Application(
        device=preferred_test_device(),
        version="v2.0",
        no_compile=True,
        dev_mode=False,
        contrast=True,
    )

    output_path = tmp_path / "contrast_synthseg.nii.gz"
    brain_mask_output_path = tmp_path / "contrast_synthstrip.nii.gz"
    results: list[dict] = []
    app.run(
        input_paths=(TEST_RES_DIR / "spgr_unstrip.nii.gz").as_posix(),
        segmentation_out=output_path.as_posix(),
        brain_mask_out=brain_mask_output_path.as_posix(),
        use_prog_bar=False,
        callback=results.append,
    )

    assert output_path.exists()
    assert len(results) == 1
    result = results[0]
    assert result["success"]
    assert "contrast_probability" in result
    assert "is_contrast" in result
    assert isinstance(result["contrast_probability"], float)
    assert 0.0 <= result["contrast_probability"] <= 1.0
    assert result["is_contrast"] == (result["contrast_probability"] >= 0.5)
