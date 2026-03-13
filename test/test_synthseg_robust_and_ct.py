from __future__ import annotations

import csv
import shutil
from pathlib import Path
from types import MethodType
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
import torch

from brain_segmentation_tools.model_manager import ModelManager
from brain_segmentation_tools.preprocessing import clip_ct_intensity

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_RES_DIR = REPO_ROOT / "test" / "res"
ORACLE_DIR = TEST_RES_DIR / "oracle"
CONVERTED_MODELS_DIR = REPO_ROOT / "build" / "converted_models"


SEGMENTATION_MIN_VOXEL_ACCURACY = 0.9985
SEGMENTATION_MIN_FOREGROUND_DICE = 0.999


def _find_model(pattern: str) -> Path | None:
    matches = sorted(CONVERTED_MODELS_DIR.glob(pattern))
    return matches[0] if matches else None


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cast(nib.Nifti1Image, nib.load(path.as_posix()))
    data = np.asanyarray(image.dataobj)
    return data, image.affine


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a_fg = a > 0
    b_fg = b > 0
    a_sum = int(a_fg.sum())
    b_sum = int(b_fg.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    intersection = int(np.logical_and(a_fg, b_fg).sum())
    return 2.0 * intersection / (a_sum + b_sum)


def _prepare_local_models(cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    segmentation_model = _find_model("synthseg_segmentation_2.0.*.pt")
    if segmentation_model is None:
        pytest.skip("Missing converted SynthSeg segmentation 2.0 model")
    parcellation_model = _find_model("synthseg_parcellation_2.0.*.pt")
    if parcellation_model is None:
        pytest.skip("Missing converted SynthSeg parcellation 2.0 model")

    local_segmentation = cache_dir / "synthseg_segmentation_2.0.pt"
    local_parcellation = cache_dir / "synthseg_parcellation_2.0.pt"
    shutil.copy2(segmentation_model, local_segmentation)
    shutil.copy2(parcellation_model, local_parcellation)

    local_robust = cache_dir / "synthseg_segmentation_robust_2.0.pt"
    robust_model = _find_model("synthseg_segmentation_robust_2.0.*.pt")
    if robust_model is not None:
        shutil.copy2(robust_model, local_robust)
    else:
        manager = ModelManager(dev_mode=True)
        try:
            manager.convert_h5_to_pt(
                model_name="synthseg",
                model_type="segmentation_robust",
                version="2.0",
                output_path=local_robust,
            )
        except FileNotFoundError:
            pytest.skip("Missing local FreeSurfer robust h5 model to convert")

    local_qc = cache_dir / "synthseg_qc_2.0.pt"
    qc_model = _find_model("synthseg_qc_2.0.*.pt")
    if qc_model is not None:
        shutil.copy2(qc_model, local_qc)
    else:
        manager = ModelManager(dev_mode=True)
        try:
            manager.convert_h5_to_pt(
                model_name="synthseg",
                model_type="qc",
                version="2.0",
                output_path=local_qc,
            )
        except FileNotFoundError:
            pytest.skip("Missing local FreeSurfer QC h5 model to convert")

    return {
        "segmentation": local_segmentation,
        "segmentation_robust": local_robust,
        "parcellation": local_parcellation,
        "qc": local_qc,
    }


def _read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def _assert_qc_csv_matches_oracle(
    actual_path: Path, oracle_path: Path, *, atol: float = 0.005
) -> None:
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
    tmp_path: Path, robust: int, parcellation: int
) -> None:
    pytest.importorskip("cupy")
    pytest.importorskip("cucim")
    pytest.importorskip("monai")

    from brain_segmentation_tools.app import Application

    local_models = _prepare_local_models(tmp_path / "model_cache")

    app = Application(
        device="cuda" if torch.cuda.is_available() else "cpu",
        version="v2.0",
        parcellation=bool(parcellation),
        robust=bool(robust),
        qc=(
            tmp_path / f"synthseg_robust_{robust}_parc_{parcellation}_qc.csv"
        ).as_posix(),
        no_compile=True,
        dev_mode=False,
        crop_segmentation_input_to_brain_mask=False,
    )

    def _local_model_path(
        self,
        *,
        model_name: str,
        model_type: str,
        version: str,
        allow_h5_in_dev: bool = True,
    ) -> Path:
        del self, allow_h5_in_dev
        clean_version = str(version).removeprefix("v")
        if (
            model_name == "synthseg"
            and model_type == "segmentation"
            and clean_version == "2.0"
        ):
            return local_models["segmentation"]
        if (
            model_name == "synthseg"
            and model_type == "segmentation_robust"
            and clean_version == "2.0"
        ):
            return local_models["segmentation_robust"]
        if (
            model_name == "synthseg"
            and model_type == "parcellation"
            and clean_version == "2.0"
        ):
            return local_models["parcellation"]
        if model_name == "synthseg" and model_type == "qc" and clean_version == "2.0":
            return local_models["qc"]
        raise KeyError(
            f"No local model mapping for {model_name}:{model_type}:{clean_version}"
        )

    cast(Any, app.model_manager).get_model_path = MethodType(
        _local_model_path, app.model_manager
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

    oracle_path = (
        ORACLE_DIR / f"synthseg_oracle_robust_{robust}_parc_{parcellation}.nii.gz"
    )
    expected_data, expected_affine = _load_nifti(oracle_path)
    actual_data, actual_affine = _load_nifti(output_path)

    assert actual_data.shape == expected_data.shape
    assert np.allclose(actual_affine, expected_affine, atol=1e-5)

    actual_labels = actual_data.astype(np.int32)
    expected_labels = expected_data.astype(np.int32)
    voxel_accuracy = float((actual_labels == expected_labels).mean())
    foreground_dice = _dice(actual_labels, expected_labels)
    print(
        f"Combo robust={robust} parc={parcellation}: "
        f"voxel_accuracy={voxel_accuracy:.6f}, foreground_dice={foreground_dice:.6f}"
    )
    assert voxel_accuracy >= SEGMENTATION_MIN_VOXEL_ACCURACY
    assert foreground_dice >= SEGMENTATION_MIN_FOREGROUND_DICE

    oracle_qc_path = (
        ORACLE_DIR / f"synthseg_oracle_robust_{robust}_parc_{parcellation}_qc.csv"
    )
    _assert_qc_csv_matches_oracle(qc_output_path, oracle_qc_path)


def test_ct_clipping_matches_freesurfer_range() -> None:
    values = torch.tensor([-200.0, 0.0, 20.0, 80.0, 120.0], dtype=torch.float32)
    clipped = clip_ct_intensity(values)
    expected = torch.tensor([0.0, 0.0, 20.0, 80.0, 80.0], dtype=torch.float32)
    assert torch.equal(clipped, expected)


def test_model_manager_has_robust_synthseg_spec() -> None:
    spec = ModelManager(dev_mode=True).get_spec(
        model_name="synthseg", model_type="segmentation_robust", version="2.0"
    )
    assert spec.freesurfer_h5 == "mri_synthseg/synthseg_robust_2.0.h5"


def test_model_manager_has_qc_synthseg_spec() -> None:
    spec = ModelManager(dev_mode=True).get_spec(
        model_name="synthseg", model_type="qc", version="2.0"
    )
    assert spec.freesurfer_h5 == "mri_synthseg/synthseg_qc_2.0.h5"
