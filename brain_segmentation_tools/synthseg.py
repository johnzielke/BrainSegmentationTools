import os

import torch
from monai.networks.layers import GaussianFilter

from brain_segmentation_tools import utils
from brain_segmentation_tools.unet_pytorch import UNet

torch._dynamo.config.capture_scalar_outputs = True

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class Synthseg(torch.nn.Module):
    FLIP_SPATIAL_AXIS = 0
    N_LEVELS = 5
    SYNTHSEG_DIVISIBLE_K = 2**N_LEVELS

    def __init__(
        self,
        *,
        model_file_segmentation,
        model_file_parcellation,
        model_file_qc,
        labels_segmentation: list[int],
        labels_denoiser,
        labels_parcellation,
        labels_qc,
        robust,
        do_parcellation,
        do_qc,
        flip_indices: dict[int, int] | None = None,
    ):
        super().__init__()

        self.model_file_segmentation = model_file_segmentation
        self.model_file_parcellation = model_file_parcellation
        self.model_file_qc = model_file_qc
        self.labels_segmentation = labels_segmentation
        self.labels_denoiser = labels_denoiser
        self.labels_parcellation = labels_parcellation
        self.labels_qc = labels_qc
        self.flip_indices = flip_indices
        if self.flip_indices is not None and not isinstance(self.flip_indices, dict):
            raise ValueError()
        self.robust = robust
        self.do_parcellation = do_parcellation
        self.do_qc = do_qc
        self.n_labels_seg = len(labels_segmentation)

        if robust:
            raise NotImplementedError()
        else:
            self.segmentation_model = UNet(
                nb_features=24,
                in_channels=1,
                nb_levels=5,
                conv_size=3,
                nb_labels=self.n_labels_seg,
                feat_mult=2,
                activation="elu",
                nb_conv_per_level=2,
                batch_norm=True,
            )
            self.segmentation_model.load_weights(
                self.model_file_segmentation, prefix="unet"
            )

            self.segmentation_gaussian_filter = GaussianFilter(
                spatial_dims=3, sigma=0.5, approx="sampled"
            )
        if do_parcellation:
            n_labels_parcellation = len(labels_parcellation)
            self.parcellation_model = UNet(
                nb_features=24,
                in_channels=3,
                nb_levels=5,
                conv_size=3,
                nb_labels=n_labels_parcellation,
                feat_mult=2,
                activation="elu",
                nb_conv_per_level=2,
                batch_norm=True,
            )
            self.parcellation_model.load_weights(
                self.model_file_parcellation, prefix="unet_parc"
            )
            self.parcellation_gaussian_filter = GaussianFilter(
                spatial_dims=3, sigma=0.5, approx="sampled"
            )
        if do_qc:
            raise NotImplementedError()
        self.eval()

    def _prepare_parcellation_input(self, *, image, segmentation):
        argmax = torch.argmax(segmentation, dim=1, keepdim=True)
        mask = (argmax == self.labels_segmentation.index(3)) | (
            argmax == self.labels_segmentation.index(42)
        )
        return torch.cat([image, ~mask, mask], dim=1)

    @utils.predict_with_padding(SYNTHSEG_DIVISIBLE_K, multiple_returns=True)
    def forward(self, image: torch.Tensor):
        n_labels_parcellation = len(self.labels_parcellation)
        segmentation = self.segmentation_model(image)
        segmentation = self.segmentation_gaussian_filter(segmentation)
        if self.flip_indices is not None:
            segmentation_flipped = self.segmentation_model(
                image.flip([self.FLIP_SPATIAL_AXIS + 2])
            )
            segmentation_flipped = torch.flip(
                segmentation_flipped, [self.FLIP_SPATIAL_AXIS + 2]
            )
            segmentation_flipped = self.segmentation_gaussian_filter(
                segmentation_flipped
            )
            rearranged_tensor = []
            for i in range(self.n_labels_seg):
                if i in self.flip_indices:
                    rearranged_tensor.append(
                        segmentation_flipped[:, self.flip_indices[i], ...]
                    )
                else:
                    rearranged_tensor.append(segmentation_flipped[:, i, ...])
            segmentation += torch.stack(rearranged_tensor, dim=1)
            segmentation *= 0.5

        if self.do_parcellation:
            # if all background, short circuit and return empty parcellation
            # TODO Removed shortcut
            if False:
                parcellation = torch.zeros(
                    image.shape[0],
                    n_labels_parcellation,
                    *image.shape[2:],
                    device=image.device,
                )
                parcellation[:, 0, ...] = 1
            else:
                parcellation = self.parcellation_model(
                    self._prepare_parcellation_input(
                        image=image, segmentation=segmentation
                    )
                )
                parcellation = self.parcellation_gaussian_filter(parcellation)
        else:
            parcellation = None
        if self.do_qc:
            raise NotImplementedError()

        return segmentation, parcellation
