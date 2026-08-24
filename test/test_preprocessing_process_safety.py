from __future__ import annotations

import io
from multiprocessing.reduction import ForkingPickler

import pytest
import torch

from brain_segmentation_tools import preprocessing
from brain_segmentation_tools.synthseg import Synthseg
from test.helpers import TEST_RES_DIR

# Keys the preprocessing pipeline can emit; each must be CPU-resident so DataLoader worker
# processes can pickle the batch back to the main process.
_OUTPUT_KEYS = ["image", "image_strip", "image_contrast"]


def _run_pre_transforms_on_cuda() -> dict:
    transform = preprocessing.get_pre_transforms(
        synthseg_divisible_k=Synthseg.SYNTHSEG_DIVISIBLE_K,
        device="cuda",
        synthstrip=True,
        synthseg=True,
        contrast_prediction=True,
    )
    return transform({"image": (TEST_RES_DIR / "spgr_unstrip.nii.gz").as_posix()})


def test_pre_transforms_emit_cpu_tensors_even_when_device_is_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    output = _run_pre_transforms_on_cuda()

    for key in _OUTPUT_KEYS:
        assert key in output, f"expected preprocessing to produce '{key}'"
        assert not output[key].is_cuda, f"'{key}' must be on CPU to cross a process boundary"


def test_pre_transforms_output_survives_forking_pickler() -> None:
    """Regression for: NotImplementedError("sharing CUDA metatensor across processes not implemented").

    DataLoader worker processes serialize each produced sample with ForkingPickler to return it to
    the main process. MONAI refuses to pickle CUDA MetaTensors, so preprocessing must yield CPU
    tensors even when a CUDA device is requested for GPU-accelerated transforms.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    output = _run_pre_transforms_on_cuda()

    for key in _OUTPUT_KEYS:
        buffer = io.BytesIO()
        # This is exactly what a process-based DataLoader worker does; it must not raise.
        ForkingPickler(buffer, -1).dump(output[key])
        assert buffer.getbuffer().nbytes > 0
