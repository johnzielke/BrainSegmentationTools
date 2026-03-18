from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from brain_segmentation_tools import postprocessing
from test.helpers import patch_single_worker_dataloader


def _require_multi_gpu_cuda_backend():
    pytest.importorskip("monai")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("cucim.skimage.measure")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.device_count() < 2:
        pytest.skip("This regression test requires at least two CUDA devices")

    return cp


class _FakeSynthsegModel(torch.nn.Module):
    def forward(self, image_tensor):
        batch_size, _channels, height, width, depth = image_tensor.shape
        segmentation = torch.full(
            (batch_size, 3, height, width, depth),
            0.05,
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )
        segmentation[:, 1] = 0.9
        return segmentation, None, None


def test_to_backend_array_uses_tensor_cuda_device_instead_of_cupy_current_device():
    cp = _require_multi_gpu_cuda_backend()

    previous_device = cp.cuda.runtime.getDevice()
    target_device = torch.device("cuda:1")

    try:
        cp.cuda.Device(0).use()
        tensor = torch.ones((1, 4, 4, 4), device=target_device, dtype=torch.float16)

        backend_array = postprocessing._to_backend_array(tensor)

        assert backend_array.device.id == target_device.index
        assert cp.cuda.runtime.getDevice() == 0
    finally:
        cp.cuda.Device(previous_device).use()


def test_pipeline_postprocessing_runs_on_selected_cuda_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = _require_multi_gpu_cuda_backend()

    from brain_segmentation_tools.app import Application

    patch_single_worker_dataloader(monkeypatch)

    input_path = tmp_path / "tiny_input.nii.gz"
    output_path = tmp_path / "tiny_output.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4, dtype=np.float32)),
        input_path,
    )

    target_device = torch.device("cuda:1")
    app = Application(
        device=str(target_device),
        version="v2.0",
        no_compile=True,
        dev_mode=False,
        crop_segmentation_input_to_brain_mask=False,
    )
    app._synthseg_model = _FakeSynthsegModel()  # type: ignore
    app.topology_classes = np.array([0, 1, 2], dtype=np.int32)
    app.labels_segmentation = np.array([0, 1, 2], dtype=np.int32)
    app.labels_parcellation = np.asarray([], dtype=np.int32)

    seen_backend_devices: list[int] = []
    seen_current_devices: list[int] = []
    original_backend_argmax = postprocessing._backend_argmax

    def _record_backend_argmax(array):
        seen_backend_devices.append(array.device.id)
        seen_current_devices.append(cp.cuda.runtime.getDevice())
        return original_backend_argmax(array)

    monkeypatch.setattr(postprocessing, "_backend_argmax", _record_backend_argmax)

    previous_device = cp.cuda.runtime.getDevice()
    try:
        cp.cuda.Device(0).use()

        app.run(
            input_paths=input_path.as_posix(),
            segmentation_out=output_path.as_posix(),
            use_prog_bar=False,
        )

        assert output_path.exists()
        assert seen_backend_devices
        assert all(device_id == target_device.index for device_id in seen_backend_devices)
        assert seen_current_devices
        assert all(device_id == target_device.index for device_id in seen_current_devices)
        assert cp.cuda.runtime.getDevice() == 0
    finally:
        cp.cuda.Device(previous_device).use()
