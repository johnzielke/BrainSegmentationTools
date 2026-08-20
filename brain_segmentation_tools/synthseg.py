import os

import torch
import torch._dynamo
import torch._dynamo.config
import torch.nn.functional as F
from monai.networks.layers import GaussianFilter

from brain_segmentation_tools import utils
from brain_segmentation_tools.qc_model import QCSynthSegRegressor
from brain_segmentation_tools.unet_pytorch import UNet, drop_unknown_state_dict_keys

torch._dynamo.config.capture_scalar_outputs = True  # ty: ignore[invalid-assignment]

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
            if labels_denoiser is None:
                raise ValueError("labels_denoiser is required when robust=True")
            self.n_labels_denoiser = len(labels_denoiser)
            self.segmentation_model_stage1 = UNet(
                nb_features=24,
                in_channels=1,
                nb_levels=5,
                conv_size=3,
                nb_labels=self.n_labels_denoiser,
                feat_mult=2,
                activation="elu",
                nb_conv_per_level=2,
                batch_norm=True,
            )
            self.segmentation_model_denoiser = UNet(
                nb_features=16,
                in_channels=self.n_labels_denoiser,
                nb_levels=5,
                conv_size=5,
                nb_labels=self.n_labels_denoiser,
                feat_mult=2,
                activation="elu",
                nb_conv_per_level=2,
                skip_n_concatenations=2,
                batch_norm=True,
            )
            self.segmentation_model_stage2 = UNet(
                nb_features=24,
                in_channels=1 + self.n_labels_denoiser,
                nb_levels=5,
                conv_size=3,
                nb_labels=self.n_labels_seg,
                feat_mult=2,
                activation="elu",
                nb_conv_per_level=2,
                batch_norm=True,
            )
            self._load_robust_weights(self.model_file_segmentation)
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
            self.segmentation_model.load_weights(self.model_file_segmentation, prefix="unet")

            self.segmentation_gaussian_filter = GaussianFilter(spatial_dims=3, sigma=0.5, approx="sampled")
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
            self.parcellation_model.load_weights(self.model_file_parcellation, prefix="unet_parc")
            self.parcellation_gaussian_filter = GaussianFilter(spatial_dims=3, sigma=0.5, approx="sampled")
        if do_qc:
            if model_file_qc is None:
                raise ValueError("model_file_qc is required when do_qc=True")
            if labels_qc is None:
                raise ValueError("labels_qc is required when do_qc=True")
            self.qc_model = QCSynthSegRegressor(
                labels_segmentation=labels_segmentation,
                labels_qc=labels_qc,
            )
            self.qc_model.load_weights(self.model_file_qc)
        self.eval()

    def _load_robust_weights(self, model_path):
        model_path = str(model_path)
        if model_path.endswith(".h5"):
            self.segmentation_model_stage1.load_weights(model_path, prefix="unet")
            self.segmentation_model_denoiser.load_weights(model_path, prefix="l2l")
            self.segmentation_model_stage2.load_weights(model_path, prefix="unet2")
            return
        if not model_path.endswith(".pt"):
            raise ValueError(f"Unsupported robust model format: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if (
            isinstance(state_dict, dict)
            and "segmentation_model_stage1" in state_dict
            and "segmentation_model_denoiser" in state_dict
            and "segmentation_model_stage2" in state_dict
        ):
            self.segmentation_model_stage1.load_state_dict(
                drop_unknown_state_dict_keys(self.segmentation_model_stage1, state_dict["segmentation_model_stage1"])
            )
            self.segmentation_model_denoiser.load_state_dict(
                drop_unknown_state_dict_keys(
                    self.segmentation_model_denoiser, state_dict["segmentation_model_denoiser"]
                )
            )
            self.segmentation_model_stage2.load_state_dict(
                drop_unknown_state_dict_keys(self.segmentation_model_stage2, state_dict["segmentation_model_stage2"])
            )
            return

        if isinstance(state_dict, dict):
            stage1_state_dict = {
                k.removeprefix("segmentation_model_stage1."): v
                for k, v in state_dict.items()
                if k.startswith("segmentation_model_stage1.")
            }
            denoiser_state_dict = {
                k.removeprefix("segmentation_model_denoiser."): v
                for k, v in state_dict.items()
                if k.startswith("segmentation_model_denoiser.")
            }
            stage2_state_dict = {
                k.removeprefix("segmentation_model_stage2."): v
                for k, v in state_dict.items()
                if k.startswith("segmentation_model_stage2.")
            }
            if stage1_state_dict and denoiser_state_dict and stage2_state_dict:
                self.segmentation_model_stage1.load_state_dict(
                    drop_unknown_state_dict_keys(self.segmentation_model_stage1, stage1_state_dict)
                )
                self.segmentation_model_denoiser.load_state_dict(
                    drop_unknown_state_dict_keys(self.segmentation_model_denoiser, denoiser_state_dict)
                )
                self.segmentation_model_stage2.load_state_dict(
                    drop_unknown_state_dict_keys(self.segmentation_model_stage2, stage2_state_dict)
                )
                return

        raise ValueError(
            f"Unsupported robust checkpoint format for {model_path}: expected stage-wise state_dict entries"
        )

    def _prepare_parcellation_input(self, *, image, segmentation):
        argmax = torch.argmax(segmentation, dim=1, keepdim=True)
        mask = (argmax == self.labels_segmentation.index(3)) | (argmax == self.labels_segmentation.index(42))
        return torch.cat([image, ~mask, mask], dim=1)

    @utils.predict_with_padding(SYNTHSEG_DIVISIBLE_K, multiple_returns=True)
    def forward(self, image: torch.Tensor):
        n_labels_parcellation = len(self.labels_parcellation)
        if self.robust:
            segmentation = self.segmentation_model_stage1(image)
            segmentation = F.one_hot(torch.argmax(segmentation, dim=1), num_classes=self.n_labels_denoiser)
            segmentation = segmentation.permute(0, 4, 1, 2, 3).to(dtype=image.dtype, device=image.device)
            segmentation = self.segmentation_model_denoiser(segmentation)
            segmentation = F.one_hot(torch.argmax(segmentation, dim=1), num_classes=self.n_labels_denoiser)
            segmentation = segmentation.permute(0, 4, 1, 2, 3).to(dtype=image.dtype, device=image.device)
            segmentation = self.segmentation_model_stage2(torch.cat([image, segmentation], dim=1))
        else:
            segmentation = self.segmentation_model(image)
            segmentation = self.segmentation_gaussian_filter(segmentation)
            if self.flip_indices is not None:
                segmentation_flipped = self.segmentation_model(image.flip([self.FLIP_SPATIAL_AXIS + 2]))
                segmentation_flipped = torch.flip(segmentation_flipped, [self.FLIP_SPATIAL_AXIS + 2])
                segmentation_flipped = self.segmentation_gaussian_filter(segmentation_flipped)
                rearranged_tensor = []
                for i in range(self.n_labels_seg):
                    if i in self.flip_indices:
                        rearranged_tensor.append(segmentation_flipped[:, self.flip_indices[i], ...])
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
                    self._prepare_parcellation_input(image=image, segmentation=segmentation)
                )
                parcellation = self.parcellation_gaussian_filter(parcellation)
        else:
            parcellation = None
        qc_scores = None
        if self.do_qc:
            qc_scores = self.qc_model.predict_scores_from_segmentation(segmentation)

        return segmentation, parcellation, qc_scores
