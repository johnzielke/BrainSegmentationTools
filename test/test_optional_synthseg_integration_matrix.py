from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter

import nibabel as nib
import numpy as np
import pytest
import torch

from brain_segmentation_tools.model_manager import ModelManager
from test.helpers import (
    ORACLE_DIR,
    REPO_ROOT,
    TEST_RES_DIR,
    dice,
    install_test_models,
    load_nifti,
    patch_single_worker_dataloader,
    preferred_test_device,
)

pytestmark = pytest.mark.slow

ZERO_INPUT_SHAPE = (64, 64, 64)
ZERO_INPUT_AFFINE = np.array(
    [
        [1.0, 0.0, 0.0, -10.0],
        [0.0, 1.0, 0.0, -20.0],
        [0.0, 0.0, 1.0, -30.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
MIN_ORACLE_VOXEL_ACCURACY = 0.985
MIN_ORACLE_FOREGROUND_DICE = 0.985
BENCHMARK_WARMUP_RUNS = 2
BENCHMARK_TIMED_RUNS = 10
_UNSUPPORTED_COMBO_CACHE: dict[tuple[str, bool], str | None] = {}


@dataclass
class RunResult:
    output_path: Path
    actual_data: np.ndarray
    actual_affine: np.ndarray
    first_inference_seconds: float
    benchmark_average_inference_seconds: float | None


@pytest.fixture(scope="session")
def synthseg_integration_model_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cache_dir = tmp_path_factory.mktemp("optional_synthseg_integration_model_cache")
    install_test_models(
        cache_dir,
        [
            ("synthseg", "segmentation", "2.0"),
            ("synthseg", "segmentation_robust", "2.0"),
            ("synthseg", "parcellation", "2.0"),
        ],
    )
    return cache_dir


@pytest.fixture(scope="session")
def zero_input_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("optional_zero_input") / "zero_input.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(ZERO_INPUT_SHAPE, dtype=np.float32), ZERO_INPUT_AFFINE), path)
    return path


@pytest.fixture(scope="session")
def benchmark_run_options_enabled() -> bool:
    value = os.environ.get("BENCHMARK_RUN_OPTIONS", "")
    return value.lower() in {"1", "true", "yes", "on"}


@pytest.fixture(scope="session")
def comparison_matrix_path() -> Path:
    device_name = preferred_test_device().split(":", 1)[0]
    comparison_matrix_path = REPO_ROOT / "build" / f"runoptions_comparison_{device_name}.csv"
    comparison_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    if comparison_matrix_path.exists():
        comparison_matrix_path.unlink()

    with open(comparison_matrix_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_kind",
                "robust",
                "no_compile",
                "dtype",
                "voxel_accuracy_error",
                "foreground_dice_error",
                "first_inference_seconds",
                "benchmark_average_inference_seconds_10_iter",
            ],
        )
        writer.writeheader()

    return comparison_matrix_path


def _unsupported_combo_reason(dtype_name: str, no_compile: bool) -> str | None:
    cache_key = (dtype_name, no_compile)
    if cache_key in _UNSUPPORTED_COMBO_CACHE:
        return _UNSUPPORTED_COMBO_CACHE[cache_key]

    dtype = getattr(torch, dtype_name)
    device = preferred_test_device()
    if not no_compile and not hasattr(torch, "compile"):
        reason = "torch.compile is unavailable in this torch build"
        _UNSUPPORTED_COMBO_CACHE[cache_key] = reason
        return reason

    try:
        model = torch.nn.Conv3d(1, 1, kernel_size=3, padding=1).to(device=device, dtype=dtype)
        sample = torch.zeros((1, 1, 8, 8, 8), device=device, dtype=dtype)
        if not no_compile:
            model = torch.compile(model)
        model(sample)
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        reason = f"{device} does not support dtype={dtype_name} with no_compile={no_compile}: {exc}"
        _UNSUPPORTED_COMBO_CACHE[cache_key] = reason
        return reason

    _UNSUPPORTED_COMBO_CACHE[cache_key] = None
    return None


def _skip_if_unsupported(dtype_name: str, no_compile: bool) -> None:
    reason = _unsupported_combo_reason(dtype_name, no_compile)
    if reason is not None:
        pytest.skip(reason)


def _append_comparison_row(
    *,
    comparison_matrix_path: Path,
    input_kind: str,
    robust: bool,
    no_compile: bool,
    dtype_name: str,
    voxel_accuracy_error: float,
    foreground_dice_error: float,
    first_inference_seconds: float,
    benchmark_average_inference_seconds: float | None,
) -> None:
    with open(comparison_matrix_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_kind",
                "robust",
                "no_compile",
                "dtype",
                "voxel_accuracy_error",
                "foreground_dice_error",
                "first_inference_seconds",
                "benchmark_average_inference_seconds_10_iter",
            ],
        )
        writer.writerow(
            {
                "input_kind": input_kind,
                "robust": int(robust),
                "no_compile": int(no_compile),
                "dtype": dtype_name,
                "voxel_accuracy_error": f"{voxel_accuracy_error:.8f}",
                "foreground_dice_error": f"{foreground_dice_error:.8f}",
                "first_inference_seconds": f"{first_inference_seconds:.8f}",
                "benchmark_average_inference_seconds_10_iter": (
                    f"{benchmark_average_inference_seconds:.8f}"
                    if benchmark_average_inference_seconds is not None
                    else ""
                ),
            }
        )


def _benchmark_average_inference_time(
    *,
    app,
    input_path: Path,
    output_dir: Path,
) -> float:
    for warmup_idx in range(BENCHMARK_WARMUP_RUNS):
        app.run(
            input_paths=input_path.as_posix(),
            segmentation_out=(output_dir / f"warmup_{warmup_idx}.nii.gz").as_posix(),
            use_prog_bar=False,
        )

    durations = []
    for run_idx in range(BENCHMARK_TIMED_RUNS):
        run_output_path = output_dir / f"benchmark_{run_idx}.nii.gz"
        durations.append(
            _run_inference_timed(
                app=app,
                input_path=input_path,
                output_path=run_output_path,
            )
        )

    return float(fmean(durations))


def _synchronize_for_timing(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_inference_timed(
    *,
    app,
    input_path: Path,
    output_path: Path,
) -> float:
    _synchronize_for_timing(app.device)
    start = perf_counter()
    app.run(
        input_paths=input_path.as_posix(),
        segmentation_out=output_path.as_posix(),
        use_prog_bar=False,
    )
    _synchronize_for_timing(app.device)
    return perf_counter() - start


def _run_segmentation(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_cache_dir: Path,
    input_path: Path,
    robust: bool,
    no_compile: bool,
    dtype_name: str,
    benchmark_run_options_enabled: bool,
) -> RunResult:
    pytest.importorskip("monai")

    from brain_segmentation_tools.app import Application

    _skip_if_unsupported(dtype_name, no_compile)
    patch_single_worker_dataloader(monkeypatch)
    monkeypatch.setenv(ModelManager.MODEL_CACHE_DIR_ENV_VAR, model_cache_dir.as_posix())

    output_path = tmp_path / f"synthseg_robust_{int(robust)}_compile_{int(not no_compile)}_{dtype_name}_parc_1.nii.gz"
    app = Application(
        device=preferred_test_device(),
        version="v2.0",
        robust=robust,
        parcellation=True,
        dtype=dtype_name,  # type: ignore
        no_compile=no_compile,
        dev_mode=False,
        crop_segmentation_input_to_brain_mask=False,
    )
    first_inference_seconds = _run_inference_timed(
        app=app,
        input_path=input_path,
        output_path=output_path,
    )

    assert output_path.exists()
    actual_data, actual_affine = load_nifti(output_path)

    benchmark_average_inference_seconds = None
    if benchmark_run_options_enabled:
        benchmark_average_inference_seconds = _benchmark_average_inference_time(
            app=app,
            input_path=input_path,
            output_dir=tmp_path / "benchmark_runs",
        )

    return RunResult(
        output_path=output_path,
        actual_data=actual_data,
        actual_affine=actual_affine,
        first_inference_seconds=first_inference_seconds,
        benchmark_average_inference_seconds=benchmark_average_inference_seconds,
    )


@pytest.mark.parametrize("robust", [False, True], ids=["standard", "robust"])
@pytest.mark.parametrize("no_compile", [True, False], ids=["no_compile", "compile"])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_synthseg_oracle_input_matrix_matches_expected_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthseg_integration_model_cache_dir: Path,
    comparison_matrix_path: Path,
    benchmark_run_options_enabled: bool,
    robust: bool,
    no_compile: bool,
    dtype_name: str,
) -> None:
    run_result = _run_segmentation(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        model_cache_dir=synthseg_integration_model_cache_dir,
        input_path=TEST_RES_DIR / "spgr_unstrip.nii.gz",
        robust=robust,
        no_compile=no_compile,
        dtype_name=dtype_name,
        benchmark_run_options_enabled=benchmark_run_options_enabled,
    )

    expected_data, expected_affine = load_nifti(ORACLE_DIR / f"synthseg_oracle_robust_{int(robust)}_parc_1.nii.gz")

    assert run_result.actual_data.shape == expected_data.shape
    assert np.allclose(run_result.actual_affine, expected_affine, atol=1e-5)

    actual_labels = run_result.actual_data.astype(np.int32)
    expected_labels = expected_data.astype(np.int32)
    voxel_accuracy = float((actual_labels == expected_labels).mean())
    foreground_dice = dice(actual_labels, expected_labels)
    voxel_accuracy_error = 1.0 - voxel_accuracy
    foreground_dice_error = 1.0 - foreground_dice
    print(
        f"oracle robust={int(robust)} no_compile={int(no_compile)} dtype={dtype_name}: "
        f"voxel_accuracy={voxel_accuracy:.6f}, foreground_dice={foreground_dice:.6f}, "
        f"first_inference_seconds={run_result.first_inference_seconds:.6f}"
    )

    assert voxel_accuracy >= MIN_ORACLE_VOXEL_ACCURACY
    assert foreground_dice >= MIN_ORACLE_FOREGROUND_DICE
    _append_comparison_row(
        comparison_matrix_path=comparison_matrix_path,
        input_kind="oracle",
        robust=robust,
        no_compile=no_compile,
        dtype_name=dtype_name,
        voxel_accuracy_error=voxel_accuracy_error,
        foreground_dice_error=foreground_dice_error,
        first_inference_seconds=run_result.first_inference_seconds,
        benchmark_average_inference_seconds=run_result.benchmark_average_inference_seconds,
    )


@pytest.mark.parametrize("robust", [False, True], ids=["standard", "robust"])
@pytest.mark.parametrize("no_compile", [True, False], ids=["no_compile", "compile"])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_synthseg_zero_input_matrix_produces_empty_segmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthseg_integration_model_cache_dir: Path,
    zero_input_path: Path,
    comparison_matrix_path: Path,
    benchmark_run_options_enabled: bool,
    robust: bool,
    no_compile: bool,
    dtype_name: str,
) -> None:
    run_result = _run_segmentation(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        model_cache_dir=synthseg_integration_model_cache_dir,
        input_path=zero_input_path,
        robust=robust,
        no_compile=no_compile,
        dtype_name=dtype_name,
        benchmark_run_options_enabled=benchmark_run_options_enabled,
    )

    assert run_result.actual_data.shape == ZERO_INPUT_SHAPE
    assert np.allclose(run_result.actual_affine, ZERO_INPUT_AFFINE, atol=1e-5)

    actual_labels = run_result.actual_data.astype(np.int32)
    expected_labels = np.zeros_like(actual_labels, dtype=np.int32)
    voxel_accuracy = float((actual_labels == expected_labels).mean())
    foreground_dice = dice(actual_labels, expected_labels)
    voxel_accuracy_error = 1.0 - voxel_accuracy
    foreground_dice_error = 1.0 - foreground_dice

    assert np.allclose(run_result.actual_data, actual_labels)
    assert np.count_nonzero(actual_labels) == 0
    _append_comparison_row(
        comparison_matrix_path=comparison_matrix_path,
        input_kind="zero",
        robust=robust,
        no_compile=no_compile,
        dtype_name=dtype_name,
        voxel_accuracy_error=voxel_accuracy_error,
        foreground_dice_error=foreground_dice_error,
        first_inference_seconds=run_result.first_inference_seconds,
        benchmark_average_inference_seconds=run_result.benchmark_average_inference_seconds,
    )
