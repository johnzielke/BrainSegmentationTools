from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from brain_segmentation_tools.model_manager import ModelManager
from brain_segmentation_tools.preprocessing import clip_ct_intensity
from test.helpers import (
    ORACLE_DIR,
    TEST_RES_DIR,
    dice,
    install_test_models,
    load_nifti,
    preferred_test_device,
)

SEGMENTATION_MIN_VOXEL_ACCURACY = 0.9985
SEGMENTATION_MIN_FOREGROUND_DICE = 0.999


def _read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def _assert_qc_csv_matches_oracle(actual_path: Path, oracle_path: Path, *, atol: float = 0.005) -> None:
    actual_rows = _read_csv(actual_path)
    oracle_rows = _read_csv(oracle_path)
    assert len(actual_rows) == len(oracle_rows)
    assert actual_rows[0] == oracle_rows[0]
    for actual_row, oracle_row in zip(actual_rows[1:], oracle_rows[1:], strict=True):
        assert actual_row[0] == oracle_row[0]
        actual_scores = np.asarray(actual_row[1:], dtype=np.float32)
        oracle_scores = np.asarray(oracle_row[1:], dtype=np.float32)
        assert np.allclose(actual_scores, oracle_scores, atol=atol)


@pytest.mark.parametrize("robust,parcellation", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_synthseg_matches_oracle_for_robust_and_parcellation_combos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    robust: int,
    parcellation: int,
) -> None:
    pytest.importorskip("monai")

    from brain_segmentation_tools.app import Application

    model_cache_dir = tmp_path / "model_cache"
    monkeypatch.setenv(ModelManager.MODEL_CACHE_DIR_ENV_VAR, model_cache_dir.as_posix())
    install_test_models(
        model_cache_dir,
        [
            ("synthseg", "segmentation", "2.0"),
            ("synthseg", "segmentation_robust", "2.0"),
            ("synthseg", "parcellation", "2.0"),
            ("synthseg", "qc", "2.0"),
        ],
    )

    app = Application(
        device=preferred_test_device(),
        version="v2.0",
        parcellation=bool(parcellation),
        robust=bool(robust),
        qc=(tmp_path / f"synthseg_robust_{robust}_parc_{parcellation}_qc.csv").as_posix(),
        no_compile=True,
        dev_mode=False,
        crop_segmentation_input_to_brain_mask=False,
    )

    output_path = tmp_path / f"synthseg_robust_{robust}_parc_{parcellation}.nii.gz"
    qc_output_path = tmp_path / f"synthseg_robust_{robust}_parc_{parcellation}_qc.csv"
    app.run(
        input_paths=(TEST_RES_DIR / "spgr_unstrip.nii.gz").as_posix(),
        segmentation_out=output_path.as_posix(),
        use_prog_bar=False,
    )
    assert output_path.exists()
    assert qc_output_path.exists()

    oracle_path = ORACLE_DIR / f"synthseg_oracle_robust_{robust}_parc_{parcellation}.nii.gz"
    expected_data, expected_affine = load_nifti(oracle_path)
    actual_data, actual_affine = load_nifti(output_path)

    assert actual_data.shape == expected_data.shape
    assert np.allclose(actual_affine, expected_affine, atol=1e-5)

    actual_labels = actual_data.astype(np.int32)
    expected_labels = expected_data.astype(np.int32)
    voxel_accuracy = float((actual_labels == expected_labels).mean())
    foreground_dice = dice(actual_labels, expected_labels)
    print(
        f"Combo robust={robust} parc={parcellation}: "
        f"voxel_accuracy={voxel_accuracy:.6f}, foreground_dice={foreground_dice:.6f}"
    )
    assert voxel_accuracy >= SEGMENTATION_MIN_VOXEL_ACCURACY
    assert foreground_dice >= SEGMENTATION_MIN_FOREGROUND_DICE

    oracle_qc_path = ORACLE_DIR / f"synthseg_oracle_robust_{robust}_parc_{parcellation}_qc.csv"
    _assert_qc_csv_matches_oracle(qc_output_path, oracle_qc_path)


def test_ct_clipping_matches_freesurfer_range() -> None:
    import torch

    values = torch.tensor([-200.0, 0.0, 20.0, 80.0, 120.0], dtype=torch.float32)
    clipped = clip_ct_intensity(values)
    expected = torch.tensor([0.0, 0.0, 20.0, 80.0, 80.0], dtype=torch.float32)
    assert torch.equal(clipped, expected)
