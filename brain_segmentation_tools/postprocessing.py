from monai import transforms
import monai
from cucim.skimage.measure import label as cucim_label
from monai.data import MetaTensor
# from cucim.skimage.morphology import binary_closing
from skimage.morphology import binary_closing
import cupy as cp
import numpy as np
import torch


def undo_padding(tensor, n_levels=5):
    if tensor.ndim == 4 and tensor.affine.ndim == 3:
        tensor = tensor.clone()
        tensor.affine = tensor.affine[0]
    padd = transforms.DivisiblePad(k=2 ** n_levels)
    # tensor.applied_operations = [tensor.applied_operations[-1]]
    monai.transforms.utils.reset_ops_id(tensor)

    tensor = padd.inverse(tensor)


    return tensor

def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    components, n_components = cucim_label(mask, return_num=True)
    return components == cp.argmax(cp.bincount(components.flatten())[1:]) + 1 if n_components > 0 else mask.copy()

def generate_brain_mask(segmentation:torch.Tensor):
    current_device = segmentation.device.index
    with cp.cuda.Device(current_device):
        mask = segmentation[0] < 0.5

        mask = get_largest_connected_component(monai.utils.convert_to_cupy(mask))
        background_mask = get_largest_connected_component(~mask)
        mask[~background_mask] = True
    mask = monai.utils.convert_to_numpy(mask)
    # out = cp.zeros_like(mask)
    # print(mask.shape)
    mask = binary_closing(mask)
    return monai.utils.convert_to_tensor(mask)

def post_process_brain_mask(segmentation:torch.Tensor, border: int = 1):
    mask = segmentation[0] < border
    current_device = segmentation.device.index
    with cp.cuda.Device(current_device):
        mask = get_largest_connected_component(monai.utils.convert_to_cupy(mask.contiguous()))
        background_mask = get_largest_connected_component(~mask)
        mask[~background_mask] = True
        mask = monai.utils.convert_to_numpy(mask)
    # out = cp.zeros_like(mask)
    # print(mask.shape)
    mask = binary_closing(mask)
    return monai.utils.convert_to_tensor(mask)


def clean_and_combine_segmentations(segmentation: torch.Tensor,parcellation: torch.Tensor, topology_classes: np.ndarray,labels_segmentation,labels_parcellation):
    current_device = segmentation.device.index
    with cp.cuda.Device(current_device):
        original_segmentation = segmentation

        segmentation = monai.utils.convert_to_cupy(segmentation)
        mask = segmentation[0] < 0.75
        if not mask.any():
            return segmentation , torch.zeros(tuple(segmentation[0].shape),dtype=torch.int32, device=original_segmentation.device)
        mask = get_largest_connected_component(mask)
        segmentation[1:] *= mask[None]
        bounding_box_start, bounding_box_end = monai.transforms.utils.generate_spatial_bounding_box(monai.utils.convert_to_tensor(mask[None]),allow_smaller=True)
        # print(bounding_box_start,bounding_box_end)
        do_crop = False
        RELATIVE_CROP_TRESHOLD = 0.1
        for i in range(3):
            if bounding_box_end[i] - bounding_box_start[i] < mask.shape[i] * (1-RELATIVE_CROP_TRESHOLD):
                do_crop = True

                break
        if do_crop:
            size_bytes_before = segmentation.nbytes
            segmentation = segmentation[:,bounding_box_start[0]:bounding_box_end[0],
                                    bounding_box_start[1]:bounding_box_end[1],
                                    bounding_box_start[2]:bounding_box_end[2]]
            # print(f"Cropping to {bounding_box_start} {bounding_box_end}, numel before: {size_bytes_before}, nbytes after: {segmentation.nbytes} {segmentation.nbytes/size_bytes_before}")
        segmentation = cp.ascontiguousarray(segmentation)
        mask = segmentation > 0.25
        for topology_class in np.unique(topology_classes)[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = cp.any(segmentation[tmp_topology_indices],axis=0)
            tmp_mask = get_largest_connected_component(tmp_mask)
            for idx in tmp_topology_indices:
                segmentation[idx] *= tmp_mask
        segmentation /= segmentation.sum(axis=0)[None]
        segmentation_argmax = monai.utils.convert_to_cupy(labels_segmentation,dtype=cp.int16)[segmentation.argmax(axis=0,dtype=cp.int8)]
        if parcellation is not None:
            parcellation = monai.utils.convert_to_cupy(parcellation)
            if do_crop:
                parcellation = parcellation[:,bounding_box_start[0]:bounding_box_end[0],
                                    bounding_box_start[1]:bounding_box_end[1],
                                    bounding_box_start[2]:bounding_box_end[2]]
            mask = (segmentation_argmax == 3) | (segmentation_argmax == 42)
            parcellation[0] = ~mask
            parc_patch = monai.utils.convert_to_cupy(labels_parcellation,dtype=cp.int16)[parcellation.argmax(axis=0,dtype=cp.int8)]
            segmentation_argmax[mask] = parc_patch[mask]
        segmentation_argmax = monai.utils.convert_to_tensor(segmentation_argmax)
        #Undo crop
        if do_crop:
            original_segmentation[:,bounding_box_start[0]:bounding_box_end[0],bounding_box_start[1]:bounding_box_end[1],bounding_box_start[2]:bounding_box_end[2]] = monai.utils.convert_to_tensor(segmentation)
            original_labels = torch.zeros(original_segmentation.shape[1],original_segmentation.shape[2],original_segmentation.shape[3],dtype=torch.int32)
            original_labels[bounding_box_start[0]:bounding_box_end[0],bounding_box_start[1]:bounding_box_end[1],bounding_box_start[2]:bounding_box_end[2]] = segmentation_argmax
        else:
            original_segmentation = monai.utils.convert_to_tensor(segmentation)
            original_labels = segmentation_argmax
        return original_segmentation, original_labels
