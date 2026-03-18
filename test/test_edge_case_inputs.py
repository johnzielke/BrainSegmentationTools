from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from brain_segmentation_tools.model_manager import ModelManager
from test.helpers import (
    install_test_models,
    load_nifti,
    patch_single_worker_dataloader,
    preferred_test_device,
)

INPUT_SHAPE = (64, 64, 64)
INPUT_AFFINE = np.array(
    [
        [1.0, 0.0, 0.0, -10.0],
        [0.0, 1.0, 0.0, -20.0],
        [0.0, 0.0, 1.0, -30.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@pytest.fixture(scope="session")
def edge_case_model_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cache_dir = tmp_path_factory.mktemp("edge_case_model_cache")
    install_test_models(
        cache_dir,
        [
            ("synthseg", "segmentation", "2.0"),
            ("synthstrip", "normal", "1"),
        ],
    )
    return cache_dir


def _run_application_on_volume(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_cache_dir: Path,
    input_name: str,
    input_data: np.ndarray,
    crop_segmentation_input_to_brain_mask: bool,
):
    pytest.importorskip("monai")

    from brain_segmentation_tools.app import Application

    patch_single_worker_dataloader(monkeypatch)
    monkeypatch.setenv(ModelManager.MODEL_CACHE_DIR_ENV_VAR, model_cache_dir.as_posix())

    input_path = tmp_path / f"{input_name}.nii.gz"
    segmentation_output_path = tmp_path / f"{input_name}_segmentation.nii.gz"
    brain_mask_output_path = tmp_path / f"{input_name}_brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(input_data.astype(np.float32), INPUT_AFFINE), input_path)

    app = Application(
        device=preferred_test_device(),
        version="v2.0",
        no_compile=True,
        dev_mode=False,
        crop_segmentation_input_to_brain_mask=crop_segmentation_input_to_brain_mask,
    )
    app.run(
        input_paths=input_path.as_posix(),
        segmentation_out=segmentation_output_path.as_posix(),
        brain_mask_out=brain_mask_output_path.as_posix(),
        use_prog_bar=False,
    )

    segmentation, segmentation_affine = load_nifti(segmentation_output_path)
    brain_mask, brain_mask_affine = load_nifti(brain_mask_output_path)
    return app, segmentation, segmentation_affine, brain_mask, brain_mask_affine


def _assert_geometry_is_preserved(
    *, segmentation: np.ndarray, segmentation_affine: np.ndarray, brain_mask: np.ndarray, brain_mask_affine: np.ndarray
) -> None:
    assert segmentation.shape == INPUT_SHAPE
    assert brain_mask.shape == INPUT_SHAPE
    assert np.allclose(segmentation_affine, INPUT_AFFINE, atol=1e-5)
    assert np.allclose(brain_mask_affine, INPUT_AFFINE, atol=1e-5)


@pytest.mark.parametrize(
    "crop_segmentation_input_to_brain_mask",
    [False, True],
    ids=["full_volume", "crop_to_brain_mask"],
)
def test_run_with_empty_input_produces_empty_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edge_case_model_cache_dir: Path,
    crop_segmentation_input_to_brain_mask: bool,
) -> None:
    app, segmentation, segmentation_affine, brain_mask, brain_mask_affine = _run_application_on_volume(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        model_cache_dir=edge_case_model_cache_dir,
        input_name=f"empty_crop_{int(crop_segmentation_input_to_brain_mask)}",
        input_data=np.zeros(INPUT_SHAPE, dtype=np.float32),
        crop_segmentation_input_to_brain_mask=(crop_segmentation_input_to_brain_mask),
    )

    _assert_geometry_is_preserved(
        segmentation=segmentation,
        segmentation_affine=segmentation_affine,
        brain_mask=brain_mask,
        brain_mask_affine=brain_mask_affine,
    )

    segmentation_labels = segmentation.astype(np.int32)
    valid_labels = set(np.unique(app.labels_segmentation.astype(np.int32)).tolist()) | {0}
    assert set(np.unique(segmentation_labels).tolist()) <= valid_labels
    assert np.count_nonzero(segmentation_labels) == 0
    assert np.count_nonzero(brain_mask) == 0


def test_run_with_random_input_produces_empty_brain_mask_and_valid_segmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edge_case_model_cache_dir: Path,
) -> None:
    input_data = np.random.default_rng(0).normal(size=INPUT_SHAPE).astype(np.float32)
    app, segmentation, segmentation_affine, brain_mask, brain_mask_affine = _run_application_on_volume(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        model_cache_dir=edge_case_model_cache_dir,
        input_name="random",
        input_data=input_data,
        crop_segmentation_input_to_brain_mask=False,
    )

    _assert_geometry_is_preserved(
        segmentation=segmentation,
        segmentation_affine=segmentation_affine,
        brain_mask=brain_mask,
        brain_mask_affine=brain_mask_affine,
    )

    segmentation_labels = segmentation.astype(np.int32)
    valid_labels = set(np.unique(app.labels_segmentation.astype(np.int32)).tolist()) | {0}
    assert set(np.unique(segmentation_labels).tolist()) <= valid_labels
    assert np.allclose(segmentation, segmentation_labels)

    brain_mask_values = set(np.unique(brain_mask).tolist())
    assert brain_mask_values <= {0.0, 1.0}
    assert np.count_nonzero(brain_mask) == 0


def test_run_with_random_input_and_empty_brain_mask_crop_produces_empty_segmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edge_case_model_cache_dir: Path,
) -> None:
    input_data = np.random.default_rng(0).normal(size=INPUT_SHAPE).astype(np.float32)
    app, segmentation, segmentation_affine, brain_mask, brain_mask_affine = _run_application_on_volume(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        model_cache_dir=edge_case_model_cache_dir,
        input_name="random_cropped_to_empty_brain_mask",
        input_data=input_data,
        crop_segmentation_input_to_brain_mask=True,
    )

    _assert_geometry_is_preserved(
        segmentation=segmentation,
        segmentation_affine=segmentation_affine,
        brain_mask=brain_mask,
        brain_mask_affine=brain_mask_affine,
    )

    segmentation_labels = segmentation.astype(np.int32)
    valid_labels = set(np.unique(app.labels_segmentation.astype(np.int32)).tolist()) | {0}
    assert set(np.unique(segmentation_labels).tolist()) <= valid_labels
    assert np.count_nonzero(segmentation_labels) == 0
    assert np.count_nonzero(brain_mask) == 0
