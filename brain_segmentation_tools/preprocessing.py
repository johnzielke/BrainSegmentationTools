from monai import transforms
import monai
import cucim
import cupy as cp
import torch
from loguru import logger

class ResampleForPrediction(transforms.Transform):
    def __init__(self, key, target_pix_dim, pix_dim_tolerance=0.05,  mode='bilinear', align_corners=False):
        self.key = key
        self.target_pix_dim = torch.tensor(target_pix_dim)
        self.pix_dim_tolerance = pix_dim_tolerance
        self.mode = mode
        self.align_corners = align_corners
        self.spacing_transform = transforms.Spacing(pixdim=self.target_pix_dim, mode=self.mode, align_corners=self.align_corners)

    def __call__(self, data):
        data = dict(data)
        img = data[self.key]
        if not isinstance(img, monai.data.MetaTensor):
            raise TypeError(f"Expected {self.key} to be of type monai.data.MetaTensor, got {type(img)}")
        if not img.ndim == 4:
            raise ValueError(f"Expected {self.key} to have 4 dimensions, got {img.ndim}")
        if not img.shape[0] == 1:
            for i in range(1, img.shape[0]):
                if img[i] != img[0]:
                    raise ValueError(f"Expected {self.key} to have 1 channel, got {img.shape[0]} non-identical channels")
            logger.warning(f"Warning: {self.key} has {img.shape[0]} identical channels, using the first one.")
            img = img[0:1]
        if any(torch.tensor(img.shape[1:]) <=1):
            raise ValueError(f"Expected {self.key} to have sizes > 1, got {img.shape[1:]}")
        pixdim = torch.sqrt(torch.sum(img.affine * img.affine, axis=0))[:-1]
        if torch.any(torch.abs(pixdim - self.target_pix_dim) > self.pix_dim_tolerance):
            factor = pixdim / self.target_pix_dim
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling
            if torch.any(sigmas > 0):
                img = transforms.GaussianSmooth(sigma=sigmas)(img)
            data[self.key] = self.spacing_transform(img)
        return data


def get_pre_transforms(synthseg_divisible_k, device='cpu', synthstrip=False, synthseg=True):
    trans = [
        transforms.LoadImaged(keys=["image"]), #TODO: Handle multichannel
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ToDeviced(keys=["image"], device=device),
    ]
    if synthstrip:
        trans += [
            transforms.CopyItemsd(keys=["image"], names=["image_strip"]),
            transforms.Orientationd(keys=["image_strip"], axcodes="LIA"),
            ResampleForPrediction(key="image_strip", target_pix_dim=(1.0, 1.0, 1.0)),
            transforms.ScaleIntensityRangePercentilesd(keys=["image_strip"], lower=0, upper=99, b_min=0.0, b_max=1.0),
            # transforms.DivisiblePadd(keys=["image_strip"], k=64),
        ]
    if synthseg:
        trans += [
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            ResampleForPrediction(key="image", target_pix_dim=(1.0, 1.0, 1.0)),
            # TODO: Crop
            # TODO: Clip for CT
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0),
            # transforms.DivisiblePadd(keys=["image"], k=synthseg_divisible_k),
        ]

    return transforms.Compose(trans)
    
