from __future__ import annotations

from pathlib import Path

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

ORACLE_BRAIN_MASK_MIN_VOXEL_ACCURACY = 0.998
ORACLE_BRAIN_MASK_MIN_DICE = 0.995
WRONG_VARIANT_MAX_VOXEL_ACCURACY = 0.99
WRONG_VARIANT_MAX_DICE = 0.95


@pytest.mark.parametrize("exclude_csf", [0, 1], ids=["normal", "nocsf"])
def test_synthstrip_matches_oracle_for_model_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exclude_csf: int
) -> None:
    pytest.importorskip("monai")

    patch_single_worker_dataloader(monkeypatch)

    from brain_segmentation_tools.app import Application

    model_cache_dir = tmp_path / "model_cache"
    monkeypatch.setenv(ModelManager.MODEL_CACHE_DIR_ENV_VAR, model_cache_dir.as_posix())
    install_test_models(
        model_cache_dir,
        [
            ("synthstrip", "normal", "1"),
            ("synthstrip", "nocsf", "1"),
        ],
    )
    app = Application(
        device=preferred_test_device(),
        version="v2.0",
        no_compile=True,
        dev_mode=False,
        brain_mask_exclude_csf=bool(exclude_csf),
    )

    output_path = tmp_path / f"synthstrip_nocsf_{exclude_csf}.nii.gz"
    app.run(
        input_paths=(TEST_RES_DIR / "spgr_unstrip.nii.gz").as_posix(),
        brain_mask_out=output_path.as_posix(),
        use_prog_bar=False,
    )
    assert output_path.exists()

    oracle_path = ORACLE_DIR / f"synthstrip_oracle_nocsf_{exclude_csf}.nii.gz"
    other_oracle_path = ORACLE_DIR / f"synthstrip_oracle_nocsf_{1 - exclude_csf}.nii.gz"
    expected_data, expected_affine = load_nifti(oracle_path)
    other_data, _ = load_nifti(other_oracle_path)
    actual_data, actual_affine = load_nifti(output_path)

    assert actual_data.shape == expected_data.shape
    assert np.allclose(actual_affine, expected_affine, atol=1e-5)

    actual_mask = (actual_data > 0).astype(np.uint8)
    expected_mask = (expected_data > 0).astype(np.uint8)
    other_mask = (other_data > 0).astype(np.uint8)

    voxel_accuracy = float((actual_mask == expected_mask).mean())
    dice_score = dice(actual_mask, expected_mask)
    wrong_variant_voxel_accuracy = float((actual_mask == other_mask).mean())
    wrong_variant_dice = dice(actual_mask, other_mask)
    print(
        f"SynthStrip nocsf={exclude_csf}: "
        f"voxel_accuracy={voxel_accuracy:.6f}, "
        f"dice={dice_score:.6f}, "
        f"wrong_variant_voxel_accuracy={wrong_variant_voxel_accuracy:.6f}, "
        f"wrong_variant_dice={wrong_variant_dice:.6f}"
    )

    assert voxel_accuracy >= ORACLE_BRAIN_MASK_MIN_VOXEL_ACCURACY
    assert dice_score >= ORACLE_BRAIN_MASK_MIN_DICE
    assert wrong_variant_voxel_accuracy <= WRONG_VARIANT_MAX_VOXEL_ACCURACY
    assert wrong_variant_dice <= WRONG_VARIANT_MAX_DICE
