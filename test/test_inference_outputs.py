from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from brain_segmentation_tools.model_manager import ModelManager
from test.helpers import (
    TEST_RES_DIR,
    dice,
    install_test_models,
    load_nifti,
    preferred_test_device,
)

SEGMENTATION_MIN_VOXEL_ACCURACY = 0.9999
SEGMENTATION_MIN_FOREGROUND_DICE = 0.9999
BRAIN_MASK_MIN_VOXEL_ACCURACY = 0.999
BRAIN_MASK_MIN_DICE = 0.9999


@pytest.fixture(scope="session")
def generated_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:  # ty:ignore[invalid-return-type]
    pytest.importorskip("monai")

    from brain_segmentation_tools.app import Application

    output_dir = tmp_path_factory.mktemp("inference")
    segmentation_out = output_dir / "synthseg_out.nii.gz"
    brain_mask_out = output_dir / "synthstrip_out.nii.gz"

    model_cache_dir = tmp_path_factory.mktemp("model_cache")
    install_test_models(
        model_cache_dir,
        [
            ("synthseg", "segmentation", "2.0"),
            ("synthseg", "parcellation", "2.0"),
            ("synthstrip", "normal", "1"),
        ],
    )

    cache_var = ModelManager.MODEL_CACHE_DIR_ENV_VAR
    previous_cache_dir = os.environ.get(cache_var)
    os.environ[cache_var] = model_cache_dir.as_posix()

    try:
        app = Application(
            device=preferred_test_device(),
            version="v2.0",
            parcellation=True,
            no_compile=True,
            dev_mode=False,
            crop_segmentation_input_to_brain_mask=False,
        )

        app.run(
            input_paths=(TEST_RES_DIR / "spgr_unstrip.nii.gz").as_posix(),
            segmentation_out=segmentation_out.as_posix(),
            brain_mask_out=brain_mask_out.as_posix(),
            use_prog_bar=False,
        )

        if not segmentation_out.exists() or not brain_mask_out.exists():
            pytest.fail(
                "Inference did not produce expected output files. "
                f"segmentation_exists={segmentation_out.exists()}, "
                f"brain_mask_exists={brain_mask_out.exists()}"
            )

        yield {
            "segmentation": segmentation_out,
            "brain_mask": brain_mask_out,
        }
    finally:
        if previous_cache_dir is None:
            os.environ.pop(cache_var, None)
        else:
            os.environ[cache_var] = previous_cache_dir


def test_synthseg_output_matches_reference(generated_outputs: dict[str, Path]) -> None:
    actual_data, actual_affine = load_nifti(generated_outputs["segmentation"])
    expected_data, expected_affine = load_nifti(TEST_RES_DIR / "synthseg.nii.gz")

    assert actual_data.shape == expected_data.shape
    assert np.allclose(actual_affine, expected_affine, atol=1e-5)

    actual_labels = actual_data.astype(np.int32)
    expected_labels = expected_data.astype(np.int32)

    voxel_accuracy = float((actual_labels == expected_labels).mean())
    foreground_dice = dice(actual_labels, expected_labels)
    print(
        "SynthSeg metrics: "
        f"voxel_accuracy={voxel_accuracy:.6f}, "
        f"foreground_dice={foreground_dice:.6f}"
    )

    assert voxel_accuracy >= SEGMENTATION_MIN_VOXEL_ACCURACY, (
        f"Segmentation voxel accuracy {voxel_accuracy:.5f} is below "
        f"threshold {SEGMENTATION_MIN_VOXEL_ACCURACY:.5f}."
    )
    assert foreground_dice >= SEGMENTATION_MIN_FOREGROUND_DICE, (
        f"Segmentation foreground Dice {foreground_dice:.5f} is below "
        f"threshold {SEGMENTATION_MIN_FOREGROUND_DICE:.5f}."
    )


def test_synthstrip_output_matches_reference(
    generated_outputs: dict[str, Path],
) -> None:
    actual_data, actual_affine = load_nifti(generated_outputs["brain_mask"])
    expected_data, expected_affine = load_nifti(TEST_RES_DIR / "synthstrip.nii.gz")

    assert actual_data.shape == expected_data.shape
    assert np.allclose(actual_affine, expected_affine, atol=1e-5)

    actual_mask = (actual_data > 0).astype(np.uint8)
    expected_mask = (expected_data > 0).astype(np.uint8)

    voxel_accuracy = float((actual_mask == expected_mask).mean())
    dice_score = dice(actual_mask, expected_mask)
    print(
        f"SynthStrip metrics: voxel_accuracy={voxel_accuracy:.6f}, "
        f"dice={dice_score:.6f}"
    )

    assert voxel_accuracy >= BRAIN_MASK_MIN_VOXEL_ACCURACY, (
        f"Brain mask voxel accuracy {voxel_accuracy:.5f} is below "
        f"threshold {BRAIN_MASK_MIN_VOXEL_ACCURACY:.5f}."
    )
    assert dice_score >= BRAIN_MASK_MIN_DICE, (
        f"Brain mask Dice {dice_score:.5f} is below "
        f"threshold {BRAIN_MASK_MIN_DICE:.5f}."
    )
