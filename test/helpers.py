from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import torch

from brain_segmentation_tools.model_manager import ModelManager, ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_RES_DIR = REPO_ROOT / "test" / "res"
ORACLE_DIR = TEST_RES_DIR / "oracle"
CONVERTED_MODELS_DIR = REPO_ROOT / "build" / "converted_models"

ModelRequest = tuple[str, str, str]


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cast(nib.Nifti1Image, nib.load(path.as_posix()))
    data = np.asanyarray(image.dataobj)
    return data, image.affine


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a_fg = a > 0
    b_fg = b > 0
    a_sum = int(a_fg.sum())
    b_sum = int(b_fg.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    intersection = int(np.logical_and(a_fg, b_fg).sum())
    return 2.0 * intersection / (a_sum + b_sum)


def install_test_models(cache_dir: Path, requests: list[ModelRequest]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manager = ModelManager(dev_mode=True)

    for model_name, model_type, version in requests:
        spec = manager.get_spec(
            model_name=model_name,
            model_type=model_type,
            version=version,
        )
        target = cache_dir / spec.pt_filename
        if target.exists():
            continue

        converted_model = _find_converted_model(spec)
        if converted_model is not None:
            shutil.copy2(converted_model, target)
            continue

        if spec.framework == "torch_state_dict":
            try:
                source = manager.get_model_path(
                    model_name=model_name,
                    model_type=model_type,
                    version=version,
                    allow_h5_in_dev=False,
                )
            except RuntimeError as exc:
                pytest.skip(
                    f"Missing local or cached source for {spec.key}: {exc}",
                )
            shutil.copy2(source, target)
            continue

        try:
            manager.convert_h5_to_pt(
                model_name=model_name,
                model_type=model_type,
                version=version,
                output_path=target,
            )
        except FileNotFoundError:
            pytest.skip(f"Missing local FreeSurfer source model for {spec.key}")


def patch_single_worker_dataloader(monkeypatch) -> None:
    from monai.data import ThreadDataLoader as MonaiThreadDataLoader

    import brain_segmentation_tools.app as app_module

    def _single_worker_loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["use_thread_workers"] = False
        return MonaiThreadDataLoader(*args, **kwargs)

    monkeypatch.setattr(app_module, "ThreadDataLoader", _single_worker_loader)


def preferred_test_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        import os

        MAX_DEFAULT_THREADS = 64
        print(
            f"CUDA not available, using CPU with up to {MAX_DEFAULT_THREADS} threads or cpu limit."
            "Change the TORCH_NUM_THREADS environment variable to set a different limit. "
            f" Detected {os.cpu_count()} cores, torch using {torch.get_num_threads()}."
        )
        TORCH_NUM_THREADS = os.environ.get("TORCH_NUM_THREADS", MAX_DEFAULT_THREADS)
        num_cores = min(os.cpu_count() or 1, int(TORCH_NUM_THREADS))
        torch.set_num_threads(num_cores)

        return "cpu"


def _find_converted_model(spec: ModelSpec) -> Path | None:
    pattern = f"{Path(spec.pt_filename).stem}.*.pt"
    matches = sorted(CONVERTED_MODELS_DIR.glob(pattern))
    return matches[0] if matches else None
