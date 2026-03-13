from __future__ import annotations

import shutil
from pathlib import Path
from types import MethodType
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_RES_DIR = REPO_ROOT / "test" / "res"
CONVERTED_MODELS_DIR = REPO_ROOT / "build" / "converted_models"


SEGMENTATION_MIN_VOXEL_ACCURACY = 0.9999
SEGMENTATION_MIN_FOREGROUND_DICE = 0.9999
BRAIN_MASK_MIN_VOXEL_ACCURACY = 0.999
BRAIN_MASK_MIN_DICE = 0.9999


def _find_model(pattern: str) -> Path:
    matches = sorted(CONVERTED_MODELS_DIR.glob(pattern))
    if not matches:
        pytest.skip(f"Missing required converted model matching: {pattern}")
    return matches[0]


def _copy_required_models(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "synthseg_segmentation_2.0.pt": _find_model("synthseg_segmentation_2.0.*.pt"),
        "synthseg_parcellation_2.0.pt": _find_model("synthseg_parcellation_2.0.*.pt"),
        "synthstrip_normal_1.pt": _find_model("synthstrip_normal_1.*.pt"),
    }
    for target_name, source in required.items():
        target = cache_dir / target_name
        if target_name == "synthstrip_normal_1.pt":
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                torch.save(checkpoint["model_state_dict"], target)
                continue
        shutil.copy2(source, target)


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


@pytest.fixture(scope="session")
def generated_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    pytest.importorskip("cupy")
    pytest.importorskip("cucim")
    pytest.importorskip("monai")

    from brain_segmentation_tools.app import Application

    output_dir = tmp_path_factory.mktemp("inference")
    segmentation_out = output_dir / "synthseg_out.nii.gz"
    brain_mask_out = output_dir / "synthstrip_out.nii.gz"

    model_cache_dir = tmp_path_factory.mktemp("model_cache")
    _copy_required_models(model_cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    app = Application(
        device=device,
        version="v2.0",
        parcellation=True,
        no_compile=True,
        dev_mode=False,
        crop_segmentation_input_to_brain_mask=False,
    )
    app.model_manager.cache_dir = model_cache_dir

    local_segmentation_model = model_cache_dir / "synthseg_segmentation_2.0.pt"
    local_parcellation_model = model_cache_dir / "synthseg_parcellation_2.0.pt"
    local_synthstrip_model = model_cache_dir / "synthstrip_normal_1.pt"

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
            return local_segmentation_model
        if (
            model_name == "synthseg"
            and model_type == "parcellation"
            and clean_version == "2.0"
        ):
            return local_parcellation_model
        if (
            model_name == "synthstrip"
            and model_type == "normal"
            and clean_version == "1"
        ):
            return local_synthstrip_model
        raise KeyError(
            f"No local model mapping for {model_name}:{model_type}:{clean_version}"
        )

    cast(Any, app.model_manager).get_model_path = MethodType(
        _local_model_path, app.model_manager
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

    return {
        "segmentation": segmentation_out,
        "brain_mask": brain_mask_out,
    }


def test_synthseg_output_matches_reference(generated_outputs: dict[str, Path]) -> None:
    actual_data, actual_affine = _load_nifti(generated_outputs["segmentation"])
    expected_data, expected_affine = _load_nifti(TEST_RES_DIR / "synthseg.nii.gz")

    assert actual_data.shape == expected_data.shape
    assert np.allclose(actual_affine, expected_affine, atol=1e-5)

    actual_labels = actual_data.astype(np.int32)
    expected_labels = expected_data.astype(np.int32)

    voxel_accuracy = float((actual_labels == expected_labels).mean())
    foreground_dice = _dice(actual_labels, expected_labels)
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
    actual_data, actual_affine = _load_nifti(generated_outputs["brain_mask"])
    expected_data, expected_affine = _load_nifti(TEST_RES_DIR / "synthstrip.nii.gz")

    assert actual_data.shape == expected_data.shape
    assert np.allclose(actual_affine, expected_affine, atol=1e-5)

    actual_mask = (actual_data > 0).astype(np.uint8)
    expected_mask = (expected_data > 0).astype(np.uint8)

    voxel_accuracy = float((actual_mask == expected_mask).mean())
    dice = _dice(actual_mask, expected_mask)
    print(f"SynthStrip metrics: voxel_accuracy={voxel_accuracy:.6f}, dice={dice:.6f}")

    assert voxel_accuracy >= BRAIN_MASK_MIN_VOXEL_ACCURACY, (
        f"Brain mask voxel accuracy {voxel_accuracy:.5f} is below "
        f"threshold {BRAIN_MASK_MIN_VOXEL_ACCURACY:.5f}."
    )
    assert dice >= BRAIN_MASK_MIN_DICE, (
        f"Brain mask Dice {dice:.5f} is below threshold {BRAIN_MASK_MIN_DICE:.5f}."
    )
