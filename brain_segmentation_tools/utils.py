from pathlib import Path
import json
import numpy as np
import functools
from concurrent import futures
import queue
import monai
import traceback
import torch
import torch.nn.functional as F
import itertools
from typing import Callable
from functools import wraps

RESOURCE_PATH = Path(__file__).parent / 'res'
if not RESOURCE_PATH.exists():
    raise Exception(f'RESOURCE_PATH does not exist: {RESOURCE_PATH}')

@functools.lru_cache
def load_resource(version, model,category):
    resource_path = RESOURCE_PATH / "constants" / str(version) / f"{model}.json"
    if not resource_path.exists():
        raise Exception(f'resource_path does not exist: {resource_path}')
    with open(resource_path, 'r') as f:
        return json.load(f)[category]


@functools.lru_cache
def get_list_labels(model, version):
    """This function reads or computes a list of all label values used in a set of label maps.
    It can also sort all labels according to FreeSurfer lut.
    """

    return np.asarray(load_resource(version, model, 'labels'), dtype=np.int32)
    
@functools.lru_cache
def get_list_labels_sorted(model, version, FS_sort=False):
    """This function reads or computes a list of all label values used in a set of label maps.
    It can also sort all labels according to FreeSurfer lut.
    :param FS_sort: (optional) whether to sort label values according to the FreeSurfer classification.
    If true, the label values will be ordered as follows: neutral labels first (i.e. non-sided), left-side labels,
    and right-side labels. If FS_sort is True, this function also returns the number of neutral labels in label_list.
    :return: the label list (numpy 1d array), and the number of neutral (i.e. non-sided) labels if FS_sort is True.
    """

    label_list = get_list_labels(model, version)
    
    # sort labels in neutral/left/right according to FS labels
    n_neutral_labels = 0
    neutral_FS_labels = [0, 14, 15, 16, 21, 22, 23, 24, 72, 77, 80, 85, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                            109, 165, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
                            251, 252, 253, 254, 255, 258, 259, 260, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
                            502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 517, 530,
                            531, 532, 533, 534, 535, 536, 537]
    neutral = list()
    left = list()
    right = list()
    for la in label_list:
        if la in neutral_FS_labels:
            if la not in neutral:
                neutral.append(la)
        elif (0 < la < 14) | (16 < la < 21) | (24 < la < 40) | (135 < la < 139) | (1000 <= la <= 1035) | \
                (la == 865) | (20100 < la < 20110):
            if la not in left:
                left.append(la)
        elif (39 < la < 72) | (162 < la < 165) | (2000 <= la <= 2035) | (20000 < la < 20010) | (la == 139) | \
                (la == 866):
            if la not in right:
                right.append(la)
        else:
            raise Exception('label {} not in our current FS classification, '
                            'please update get_list_labels in utils.py'.format(la))
    label_list = np.concatenate([sorted(neutral), sorted(left), sorted(right)])
    if ((len(left) > 0) & (len(right) > 0)) | ((len(left) == 0) & (len(right) == 0)):
        n_neutral_labels = len(neutral)
    else:
        n_neutral_labels = len(label_list)

    return np.int32(label_list), n_neutral_labels
    


def get_flip_indices(labels_segmentation, n_neutral_labels):

    # get position labels
    n_sided_labels = int((len(labels_segmentation) - n_neutral_labels) / 2)
    neutral_labels = labels_segmentation[:n_neutral_labels]
    left = labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels]

    # get correspondance between labels
    lr_corresp = np.stack([labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels],
                           labels_segmentation[n_neutral_labels + n_sided_labels:]])
    lr_corresp_unique, lr_corresp_indices = np.unique(lr_corresp[0, :], return_index=True)
    lr_corresp_unique = np.stack([lr_corresp_unique, lr_corresp[1, lr_corresp_indices]])
    lr_corresp_unique = lr_corresp_unique[:, 1:] if not np.all(lr_corresp_unique[:, 0]) else lr_corresp_unique

    # get unique labels
    labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)

    # get indices of corresponding labels
    lr_indices = np.zeros_like(lr_corresp_unique)
    for i in range(lr_corresp_unique.shape[0]):
        for j, lab in enumerate(lr_corresp_unique[i]):
            lr_indices[i, j] = np.where(labels_segmentation == lab)[0]

    # build 1d vector to swap LR corresponding labels taking into account neutral labels
    flip_indices = {}
    for i in range(len(labels_segmentation)):
        if labels_segmentation[i] in neutral_labels:
            flip_indices[i] = i
        elif labels_segmentation[i] in left:
            flip_indices[i] = lr_indices[1, np.where(lr_corresp_unique[0, :] == labels_segmentation[i])].item()
        else:
            flip_indices[i] = lr_indices[0, np.where(lr_corresp_unique[1, :] == labels_segmentation[i])].item()

    return labels_segmentation, flip_indices, unique_idx

def get_model_file(model_type:str, version:str,format="h5",model_name="synthseg"):
    version = version.removeprefix("v")
    return RESOURCE_PATH / "models" / f"{model_name}_{model_type}_{version}.{format}"

class ThreadPoolExecutorWithQueueSizeLimit(futures.ThreadPoolExecutor):
    def __init__(self, maxsize=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._work_queue = queue.Queue(maxsize=maxsize)


class ErrorCatchingDataset(monai.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        



    def _transform(self, index:int):
        try:
            return self.dataset[index]
        except Exception as e:
            return {
                **self.dataset.data[index],
                "exception": "".join(line for line in traceback.format_exception(type(e), e, e.__traceback__) if not line.lstrip().startswith('File') and not line.lstrip().startswith('Traceback') and line.lstrip())
            }
        
    def __len__(self):
        return len(self.dataset)


def convert_to_meta_tensor(data: torch.Tensor,*, copy_meta_from: monai.data.MetaTensor=None) -> monai.data.MetaTensor:
    return monai.data.MetaTensor(data, affine=copy_meta_from.affine, meta=copy_meta_from.meta, applied_operations=copy_meta_from.applied_operations)



def predict_with_padding(divisible_by: int,multiple_returns=False):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(model, x: torch.Tensor, *args, **kwargs):
            pad_width = []
            for s in x.shape[2:]:
                total_pad = (divisible_by - s % divisible_by) % divisible_by
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width.append((pad_before, pad_after))
            pad_args = tuple(itertools.chain(*pad_width[::-1]))
            padded_tensor = F.pad(x, pad_args)
            out = func(model, padded_tensor, *args, **kwargs)
            unpad_slices = [slice(None), slice(None)] + [slice(s[0], -s[1] if s[1] > 0 else None) for s in pad_width]
            if multiple_returns:
                if isinstance(out, (list, tuple)):
                    unpadded_out = [o[tuple(unpad_slices)] for o in out]
                elif isinstance(out, dict):
                    unpadded_out = {k: v[tuple(unpad_slices)] if isinstance(v, torch.Tensor) else v for k, v in out.items()}
                else:
                    raise ValueError("Output type not supported for multiple returns.")
            else:
                unpadded_out = out[tuple(unpad_slices)]
            return unpadded_out
        return wrapper
    return decorator
