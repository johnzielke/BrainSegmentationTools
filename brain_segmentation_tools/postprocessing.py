from __future__ import annotations

from contextlib import nullcontext

import monai
import numpy as np
import torch
from monai import transforms
from monai.transforms.utils import generate_spatial_bounding_box, reset_ops_id
from scipy import ndimage

# from cucim.skimage.morphology import binary_closing
from skimage.morphology import binary_closing

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency for CUDA acceleration
    cp = None  # type: ignore[assignment]

try:
    from cucim.skimage.measure import label as cucim_label
except ImportError:  # pragma: no cover - optional dependency for CUDA acceleration
    cucim_label = None  # type: ignore[assignment]


def undo_padding(tensor, n_levels=5):
    if tensor.ndim == 4 and tensor.affine.ndim == 3:
        tensor = tensor.clone()
        tensor.affine = tensor.affine[0]
    padd = transforms.DivisiblePad(k=2**n_levels)
    # tensor.applied_operations = [tensor.applied_operations[-1]]
    reset_ops_id(tensor)

    tensor = padd.inverse(tensor)

    return tensor


def _is_cupy_array(array) -> bool:
    return cp is not None and isinstance(array, cp.ndarray)


def _can_use_cuda_backend(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda" and cp is not None and cucim_label is not None


def _cupy_device_context(device: torch.device):
    if cp is None or device.type != "cuda":
        return nullcontext()

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return cp.cuda.Device(device_index)


def _to_backend_array(tensor: torch.Tensor):
    if _can_use_cuda_backend(tensor):
        with _cupy_device_context(tensor.device):
            return monai.utils.convert_to_cupy(tensor)
    return monai.utils.convert_to_numpy(tensor)


def _backend_any(array, axis=0):
    if _is_cupy_array(array):
        return cp.any(array, axis=axis)
    return np.any(array, axis=axis)


def _backend_ascontiguousarray(array):
    if _is_cupy_array(array):
        return cp.ascontiguousarray(array)
    return np.ascontiguousarray(array)


def _backend_argmax(array):
    if _is_cupy_array(array):
        return array.argmax(axis=0, dtype=cp.int8)
    return array.argmax(axis=0)


def _backend_labels(
    values,
    *,
    use_cuda: bool,
    device: torch.device | None = None,
):
    if use_cuda:
        if device is None:
            raise ValueError("device must be provided when using the CUDA backend")
        with _cupy_device_context(device):
            return monai.utils.convert_to_cupy(values, dtype=np.dtype(np.int16))
    return np.asarray(values, dtype=np.int16)


def _backend_to_tensor(array, *, device: torch.device, dtype: torch.dtype | None = None):
    tensor = monai.utils.convert_to_tensor(array)
    return tensor.to(device=device, dtype=dtype if dtype is not None else tensor.dtype)


def _close_mask(mask, *, device: torch.device) -> torch.Tensor:
    mask = binary_closing(monai.utils.convert_to_numpy(mask))
    return _backend_to_tensor(mask, device=device)


def _fill_brain_mask(mask):
    mask = get_largest_connected_component(mask)
    background_mask = get_largest_connected_component(~mask)
    mask[~background_mask] = True
    return mask


def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    if _is_cupy_array(mask):
        components, n_components = cucim_label(mask, return_num=True)
        if n_components == 0:
            return mask.copy()
        component_sizes = cp.bincount(components.ravel())
        component_sizes[0] = 0
        return components == cp.argmax(component_sizes)

    mask = np.asarray(mask)
    components, n_components = ndimage.label(mask, structure=structure)
    if n_components == 0:
        return mask.copy()
    component_sizes = np.bincount(components.ravel())
    component_sizes[0] = 0
    return components == component_sizes.argmax()


def generate_brain_mask(segmentation: torch.Tensor):
    with _cupy_device_context(segmentation.device):
        mask = _to_backend_array(segmentation[0] < 0.5)
        mask = _fill_brain_mask(mask)
        return _close_mask(mask, device=segmentation.device)


def post_process_brain_mask(segmentation: torch.Tensor, border: int = 1):
    with _cupy_device_context(segmentation.device):
        mask = _to_backend_array((segmentation[0] < border).contiguous())
        mask = _fill_brain_mask(mask)
        return _close_mask(mask, device=segmentation.device)


def clean_and_combine_segmentations(
    segmentation: torch.Tensor,
    parcellation: torch.Tensor | None,
    topology_classes: np.ndarray,
    labels_segmentation,
    labels_parcellation,
):
    original_segmentation = segmentation
    use_cuda = _can_use_cuda_backend(segmentation)

    with _cupy_device_context(original_segmentation.device):
        segmentation = _to_backend_array(segmentation)
        mask = segmentation[0] < 0.75
        if not mask.any():
            return original_segmentation, torch.zeros(
                tuple(segmentation[0].shape),
                dtype=torch.int32,
                device=original_segmentation.device,
            )
        mask = get_largest_connected_component(mask)
        segmentation[1:] *= mask[None]
        bounding_box_start, bounding_box_end = generate_spatial_bounding_box(
            _backend_to_tensor(mask[None], device=torch.device("cpu")),
            allow_smaller=True,
        )
        do_crop = False
        RELATIVE_CROP_TRESHOLD = 0.1
        for i in range(3):
            if bounding_box_end[i] - bounding_box_start[i] < mask.shape[i] * (1 - RELATIVE_CROP_TRESHOLD):
                do_crop = True
                break
        if do_crop:
            segmentation = segmentation[
                :,
                bounding_box_start[0] : bounding_box_end[0],
                bounding_box_start[1] : bounding_box_end[1],
                bounding_box_start[2] : bounding_box_end[2],
            ]
        segmentation = _backend_ascontiguousarray(segmentation)
        for topology_class in np.unique(topology_classes)[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = _backend_any(segmentation[tmp_topology_indices], axis=0)
            tmp_mask = get_largest_connected_component(tmp_mask)
            for idx in tmp_topology_indices:
                segmentation[idx] *= tmp_mask
        segmentation /= segmentation.sum(axis=0)[None]
        segmentation_argmax = _backend_labels(
            labels_segmentation,
            use_cuda=use_cuda,
            device=original_segmentation.device,
        )[_backend_argmax(segmentation)]
        if parcellation is not None:
            parcellation = _to_backend_array(parcellation)
            if do_crop:
                parcellation = parcellation[
                    :,
                    bounding_box_start[0] : bounding_box_end[0],
                    bounding_box_start[1] : bounding_box_end[1],
                    bounding_box_start[2] : bounding_box_end[2],
                ]
            mask = (segmentation_argmax == 3) | (segmentation_argmax == 42)
            parcellation[0] = ~mask
            parc_patch = _backend_labels(
                labels_parcellation,
                use_cuda=use_cuda,
                device=original_segmentation.device,
            )[_backend_argmax(parcellation)]
            segmentation_argmax[mask] = parc_patch[mask]
        segmentation_argmax_tensor = _backend_to_tensor(
            segmentation_argmax,
            device=original_segmentation.device,
            dtype=torch.int32,
        )
        if do_crop:
            original_segmentation[
                :,
                bounding_box_start[0] : bounding_box_end[0],
                bounding_box_start[1] : bounding_box_end[1],
                bounding_box_start[2] : bounding_box_end[2],
            ] = _backend_to_tensor(
                segmentation,
                device=original_segmentation.device,
                dtype=original_segmentation.dtype,
            )
            original_labels = torch.zeros(
                original_segmentation.shape[1],
                original_segmentation.shape[2],
                original_segmentation.shape[3],
                dtype=torch.int32,
                device=original_segmentation.device,
            )
            original_labels[
                bounding_box_start[0] : bounding_box_end[0],
                bounding_box_start[1] : bounding_box_end[1],
                bounding_box_start[2] : bounding_box_end[2],
            ] = segmentation_argmax_tensor
        else:
            original_segmentation = _backend_to_tensor(
                segmentation,
                device=original_segmentation.device,
                dtype=original_segmentation.dtype,
            )
            original_labels = segmentation_argmax_tensor
        return original_segmentation, original_labels
