import csv
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from fire import Fire
from monai import transforms
from monai.data import Dataset, MetaTensor, ThreadDataLoader
from monai.transforms.utils import generate_spatial_bounding_box
from tqdm import tqdm

from brain_segmentation_tools import postprocessing, preprocessing, utils
from brain_segmentation_tools.model_manager import ModelManager
from brain_segmentation_tools.synthseg import Synthseg
from brain_segmentation_tools.synthstrip import StripModel

torch._dynamo.config.capture_scalar_outputs = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class Application:
    parcellation: bool = False
    robust: bool = False
    fast: bool = False
    ct: bool = False
    vol: str | None = None
    qc: str | None = None
    crop: list[int] | None = None
    version: str = "v2.0"
    dev_mode: bool | None = None
    device: str = "cuda"
    synthseg: Synthseg = field(init=False)
    flip_indices: dict[int, int] | None = field(init=False)
    n_neutral_labels: int = field(init=False)
    labels_segmentation: np.ndarray = field(init=False)
    labels_denoiser: np.ndarray = field(init=False)
    labels_parcellation: np.ndarray = field(init=False)
    labels_qc: np.ndarray = field(init=False)
    names_qc: np.ndarray = field(init=False)
    _synthseg_model: Synthseg | None = field(init=False, default=None)
    _synthstrip_model: StripModel | None = field(init=False, default=None)
    model_manager: ModelManager = field(init=False)
    topology_classes: np.ndarray = field(init=False)
    output_extension: str = ".nii.gz"
    input_file_extensions: tuple[str, ...] = (".nii", ".nii.gz")
    dtype: torch.dtype = torch.float32
    skip_existing: bool = False
    no_compile: bool = True
    brain_mask_exclude_csf: bool = False
    brain_mask_border: int = 1
    crop_segmentation_input_to_brain_mask: bool = True

    BATCH_SIZE = 1  # Prediction currently supports only batch size 1.

    def __post_init__(self):
        assert self.version in ["v1.0", "v2.0"]
        if self.robust and self.version != "v2.0":
            raise ValueError("robust mode is only available for SynthSeg v2.0")
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)
        self.model_manager = ModelManager(dev_mode=self.dev_mode)

        self.n_neutral_labels = 19 if self.version == "v2.0" else 18

        labels_segmentation = utils.get_list_labels("segmentation", self.version)
        if (not self.fast) & (not self.robust):
            self.labels_segmentation, self.flip_indices, unique_idx = (
                utils.get_flip_indices(labels_segmentation, self.n_neutral_labels)
            )
        else:
            self.labels_segmentation, unique_idx = np.unique(
                labels_segmentation, return_index=True
            )
            self.flip_indices = None
        self.labels_denoiser = (
            utils.get_list_labels("denoiser", self.version)
            if self.robust
            else np.asarray([], dtype=np.int32)
        )
        self.labels_parcellation = utils.get_list_labels("parcellation", self.version)
        self.labels_qc = utils.get_list_labels("qc", self.version)[unique_idx]
        self.names_qc = np.asarray(utils.load_resource(self.version, "qc", "names"))[
            unique_idx
        ]
        self.topology_classes = np.asarray(
            utils.load_resource(self.version, "topological", "classes")
        )[unique_idx]

    @property
    def synthseg_model(self):
        if self._synthseg_model is not None:
            return self._synthseg_model
        segmentation_model_type = (
            "segmentation_robust" if self.robust else "segmentation"
        )
        segmentation_model_file = self.model_manager.get_model_path(
            model_name="synthseg",
            model_type=segmentation_model_type,
            version=self.version,
            allow_h5_in_dev=True,
        )
        parcellation_model_file = None
        if self.parcellation:
            parcellation_model_file = self.model_manager.get_model_path(
                model_name="synthseg",
                model_type="parcellation",
                version=self.version,
                allow_h5_in_dev=True,
            )
        qc_model_file = None
        if self.qc:
            qc_model_file = self.model_manager.get_model_path(
                model_name="synthseg",
                model_type="qc",
                version=self.version,
                allow_h5_in_dev=True,
            )
        self._synthseg_model = Synthseg(
            model_file_segmentation=segmentation_model_file,
            model_file_parcellation=parcellation_model_file,
            model_file_qc=qc_model_file,
            labels_segmentation=self.labels_segmentation.tolist(),
            labels_denoiser=self.labels_denoiser.tolist() if self.robust else None,
            labels_parcellation=self.labels_parcellation.tolist(),
            labels_qc=self.labels_qc.tolist() if self.qc else None,
            robust=self.robust,
            do_parcellation=self.parcellation,
            do_qc=bool(self.qc),
            flip_indices=self.flip_indices,
        )
        self._synthseg_model = self._synthseg_model.to(self.device, dtype=self.dtype)
        self._synthseg_model.eval()
        if not self.no_compile:
            self._synthseg_model = cast(Synthseg, torch.compile(self._synthseg_model))
        return self._synthseg_model

    @property
    def synthstrip_model(self):
        if self._synthstrip_model is not None:
            return self._synthstrip_model
        synthstrip_model_file = self.model_manager.get_model_path(
            model_name="synthstrip",
            model_type="nocsf" if self.brain_mask_exclude_csf else "normal",
            version="1",
            allow_h5_in_dev=False,
        )
        self._synthstrip_model = StripModel()
        self._synthstrip_model.load_state_dict(
            torch.load(synthstrip_model_file, weights_only=True)
        )
        self._synthstrip_model = self._synthstrip_model.to(
            self.device, dtype=self.dtype
        )
        self._synthstrip_model.eval()
        if not self.no_compile:
            self._synthstrip_model = cast(
                StripModel, torch.compile(self._synthstrip_model)
            )

        return self._synthstrip_model

    @torch.inference_mode()
    def predict_synthseg_batch(self, image_tensor, brain_mask=None):
        """
        image_tensor: A batch of images to segment. Shape (B, C, H, W, D)
        brain_mask: Optional batch of brain masks used to crop inputs.
            Shape (H, W, D)
        """
        if brain_mask is not None:
            original_shape = image_tensor.shape[2:]
            bbox_start, bbox_end = generate_spatial_bounding_box(
                brain_mask[None], margin=5
            )
            image_tensor = image_tensor[
                :,
                :,
                bbox_start[0] : bbox_end[0],
                bbox_start[1] : bbox_end[1],
                bbox_start[2] : bbox_end[2],
            ]
        segmentation, parcellation, qc_scores = self.synthseg_model(
            image_tensor.to(self.device, dtype=self.dtype)
        )

        results = []
        for i in range(segmentation.shape[0]):
            seg_ = segmentation[i]
            parc_ = None
            if self.parcellation:
                parc_ = parcellation[i]
            segmentation_cleaned_posteriors, segmentation_final_labels = (
                postprocessing.clean_and_combine_segmentations(
                    segmentation=seg_.to(dtype=torch.float16),
                    parcellation=parc_.to(dtype=torch.float16)
                    if parc_ is not None
                    else None,
                    topology_classes=self.topology_classes,
                    labels_segmentation=self.labels_segmentation,
                    labels_parcellation=self.labels_parcellation,
                )
            )
            if brain_mask is not None:
                # Pad the segmentation back to the original size
                segmentation_final_labels_ = torch.zeros(
                    original_shape,
                    dtype=segmentation_final_labels.dtype,
                    device=segmentation_final_labels.device,
                )
                segmentation_final_labels_[
                    bbox_start[0] : bbox_end[0],
                    bbox_start[1] : bbox_end[1],
                    bbox_start[2] : bbox_end[2],
                ] = segmentation_final_labels
                segmentation_final_labels = segmentation_final_labels_
            row = {}
            input_image = image_tensor[i]
            if isinstance(input_image, MetaTensor):
                segmentation_final_labels = MetaTensor(
                    segmentation_final_labels, affine=input_image.affine
                )
            row["segmentation"] = segmentation_final_labels
            if qc_scores is not None:
                row["qc_scores"] = qc_scores[i]
            results.append(row)
        return results

    @staticmethod
    def _subject_id_for_qc(path: str) -> str:
        name = Path(path).name
        for suffix in [".nii.gz", ".nii", ".mgz", ".npz"]:
            if name.endswith(suffix):
                return name.removesuffix(suffix)
        return Path(name).stem

    def _qc_headers(self) -> list[str]:
        _, unique_idx = np.unique(self.labels_qc, return_index=True)
        return self.names_qc[unique_idx].tolist()[1:]

    def _write_qc_header(self, qc_output_path: Path) -> None:
        qc_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(qc_output_path, "w", newline="") as f:
            csv.writer(f).writerow(["subject", *self._qc_headers()])

    def _append_qc_row(
        self, *, qc_output_path: Path, input_image_path: str, qc_scores: torch.Tensor
    ) -> None:
        scores = np.clip(
            np.squeeze(qc_scores.detach().float().cpu().numpy())[1:], 0.0, 1.0
        )
        row = [self._subject_id_for_qc(input_image_path)] + [
            f"{score:.4f}" for score in scores
        ]
        with open(qc_output_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    @torch.inference_mode()
    def predict_synthstrip_batch(self, data):
        brain_mask = self.synthstrip_model(
            data["image_strip"].to(self.device, dtype=self.dtype)
        )
        reorient = transforms.Orientation(axcodes="RAS")
        results = []
        for i in range(brain_mask.shape[0]):
            brain_mask_ = brain_mask[i]
            brain_mask_ = reorient(brain_mask_)
            brain_mask = postprocessing.post_process_brain_mask(
                brain_mask_, border=self.brain_mask_border
            )
            row = {}
            source_image_strip = data["image_strip"][i]
            affine = None
            if isinstance(source_image_strip, MetaTensor):
                affine = source_image_strip.affine
            if isinstance(brain_mask_, MetaTensor):
                # Preserve the affine after explicit RAS reorientation.
                affine = brain_mask_.affine
            if affine is not None:
                brain_mask = MetaTensor(brain_mask, affine=affine)
            row["brain_mask"] = brain_mask
            results.append(row)
        return results

    @torch.inference_mode()
    def run(
        self,
        input_paths: str | list[str],
        *,
        segmentation_out: str | list[str] | None = None,
        brain_mask_out: str | list[str] | None = None,
        data_root: str | None = None,
        callback: Callable[[dict[str, object]], None] | None = None,
        ids: list[str] | None = None,
        use_prog_bar: bool = True,
    ):
        do_segmentation = segmentation_out is not None
        do_brain_mask = brain_mask_out is not None
        do_qc = self.qc is not None
        qc_output_path = None
        if do_qc:
            if not do_segmentation:
                raise ValueError("qc output requires segmentation_out")
            qc_output_path = Path(cast(str, self.qc))
            if qc_output_path.suffix.lower() != ".csv":
                qc_output_path = qc_output_path.with_suffix(".csv")
            self._write_qc_header(qc_output_path)
        if (
            isinstance(input_paths, str)
            and Path(input_paths).is_file()
            and input_paths.endswith(".txt")
        ):
            input_paths = list(Path(input_paths).read_text().splitlines())
        input_paths = input_paths if isinstance(input_paths, list) else [input_paths]
        segmentation_out_items: list[str | None]
        brain_mask_out_items: list[str | None]
        if data_root is not None:
            if segmentation_out is not None:
                if not isinstance(segmentation_out, str):
                    raise ValueError(
                        "segmentation_out must be a string if data_root is provided"
                    )
                segmentation_out_items = [cast(str | None, segmentation_out)] * len(
                    input_paths
                )
            else:
                segmentation_out_items = [None] * len(input_paths)
            if brain_mask_out is not None:
                if not isinstance(brain_mask_out, str):
                    raise ValueError(
                        "brain_mask_out must be a string if data_root is provided"
                    )
                brain_mask_out_items = [cast(str | None, brain_mask_out)] * len(
                    input_paths
                )
            else:
                brain_mask_out_items = [None] * len(input_paths)
        else:
            segmentation_out_items = (
                [cast(str | None, path) for path in segmentation_out]
                if isinstance(segmentation_out, list)
                else [segmentation_out]
            )
            brain_mask_out_items = (
                [cast(str | None, path) for path in brain_mask_out]
                if isinstance(brain_mask_out, list)
                else [brain_mask_out]
            )

        extended_input_paths = []
        extended_segmentation_out_paths = []
        extended_brain_mask_out_paths = []
        for input_path, brain_mask_out_path, segmentation_out_path in zip(
            input_paths, brain_mask_out_items, segmentation_out_items, strict=False
        ):
            input_path = Path(input_path)
            if input_path.is_dir():
                if (
                    segmentation_out_path is not None
                    and not Path(segmentation_out_path).is_dir()
                ):
                    raise ValueError(
                        "segmentation_out must be a directory "
                        "if input_path is a directory"
                    )
                if (
                    brain_mask_out_path is not None
                    and not Path(brain_mask_out_path).is_dir()
                ):
                    raise ValueError(
                        "brain_mask_out must be a directory "
                        "if input_path is a directory"
                    )
                for extension in self.input_file_extensions:
                    matched_files = list(input_path.rglob(f"*{extension}"))
                    extended_input_paths.extend(p for p in matched_files)
                    file_rel_paths = [
                        Path(
                            p.relative_to(input_path).as_posix().removesuffix(extension)
                            + self.output_extension
                        )
                        for p in matched_files
                    ]
                    if segmentation_out_path is not None:
                        extended_segmentation_out_paths.extend(
                            [segmentation_out_path / p for p in file_rel_paths]
                        )
                    else:
                        extended_segmentation_out_paths.extend(
                            [None] * len(file_rel_paths)
                        )
                    if brain_mask_out_path is not None:
                        extended_brain_mask_out_paths.extend(
                            [brain_mask_out_path / p for p in file_rel_paths]
                        )
                    else:
                        extended_brain_mask_out_paths.extend(
                            [None] * len(file_rel_paths)
                        )
                    print(
                        f"Found {len(matched_files)} files for {input_path} "
                        f"with extension {extension}"
                    )
            else:
                extended_input_paths.append(input_path)
                if data_root is not None:
                    rel_path = Path(
                        input_path.relative_to(data_root)
                        .as_posix()
                        .removesuffix(".nii.gz")
                        + self.output_extension
                    )
                    if segmentation_out_path is not None:
                        extended_segmentation_out_paths.append(
                            segmentation_out_path / rel_path
                        )
                    else:
                        extended_segmentation_out_paths.append(None)
                    if brain_mask_out_path is not None:
                        extended_brain_mask_out_paths.append(
                            brain_mask_out_path / rel_path
                        )
                    else:
                        extended_brain_mask_out_paths.append(None)
                else:
                    extended_segmentation_out_paths.append(
                        Path(segmentation_out_path)
                        if segmentation_out_path is not None
                        else None
                    )
                    extended_brain_mask_out_paths.append(
                        Path(brain_mask_out_path)
                        if brain_mask_out_path is not None
                        else None
                    )
        input_paths = extended_input_paths
        segmentation_out_paths = extended_segmentation_out_paths
        brain_mask_out_paths = extended_brain_mask_out_paths
        if ids is None:
            ids = [str(i) for i in range(len(input_paths))]
        elif len(ids) != len(input_paths):
            raise ValueError("ids must have the same length as input_paths")

        if len(input_paths) != len(segmentation_out_paths):
            raise ValueError("input_path and output_path must have the same length")

        if not input_paths:
            raise ValueError("No scans provided")

        existing = [Path(path).exists() for path in input_paths]
        if not all(existing):
            non_existing = [input_paths[i] for i, e in enumerate(existing) if not e]
            raise ValueError(f"input_paths {non_existing} do not exist")
        if self.skip_existing:
            existing_segmentations = [
                path is None or Path(path).exists() for path in segmentation_out_paths
            ]
            existing_brain_mask = [
                path is None or Path(path).exists() for path in brain_mask_out_paths
            ]
            existing_outputs = np.logical_and(
                existing_segmentations, existing_brain_mask
            )
            input_paths = [
                input_path
                for input_path, existing_output in zip(
                    input_paths, existing_outputs, strict=False
                )
                if not existing_output
            ]
            segmentation_out_paths = [
                output_path
                for output_path, existing_output in zip(
                    segmentation_out_paths, existing_outputs, strict=False
                )
                if not existing_output
            ]
            brain_mask_out_paths = [
                brain_mask_out_path
                for brain_mask_out_path, existing_output in zip(
                    brain_mask_out_paths, existing_outputs, strict=False
                )
                if not existing_output
            ]
            print(f"Skipping {sum(existing_outputs)} existing segmentations")
        preprocessing_transform = preprocessing.get_pre_transforms(
            synthseg_divisible_k=Synthseg.SYNTHSEG_DIVISIBLE_K,
            device=self.device,
            synthstrip=do_brain_mask,
            synthseg=do_segmentation,
            ct=self.ct,
        )
        dataset = utils.ErrorCatchingDataset(
            Dataset(
                [
                    {
                        "image": image.as_posix(),
                        "output": output.as_posix() if output is not None else "",
                        "brain_mask_output": brain_mask_out.as_posix()
                        if brain_mask_out is not None
                        else "",
                        "id": id,
                    }
                    for image, output, brain_mask_out, id in zip(
                        input_paths,
                        segmentation_out_paths,
                        brain_mask_out_paths,
                        ids,
                        strict=False,
                    )
                ],
                preprocessing_transform,
            )
        )
        assert self.BATCH_SIZE == 1, "Batch size must be 1 for the pipeline logic"
        dataloader = ThreadDataLoader(
            dataset, batch_size=self.BATCH_SIZE, use_thread_workers=True, num_workers=4
        )
        saver = transforms.SaveImage(output_ext=self.output_extension, print_log=False)
        progress: tqdm[Any] | None = tqdm(dataloader) if use_prog_bar else None
        data_iterable = progress if progress is not None else dataloader
        for data in data_iterable:
            # TODO: Use brain mask to crop segmentation input.
            if "exception" in data:
                print(
                    f"Skipping {data['image'][0]} "
                    f"due to exception: {data['exception'][0]}"
                )
                if callback is not None:
                    callback(
                        {
                            "id": data["id"][0],
                            "exception": {
                                "message": str(data["exception"][0]),
                                "type": type(data["exception"][0]).__name__,
                            },
                            "success": False,
                        }
                    )
                continue
            input_image_path = data["image"][0].meta.get("filename_or_obj")
            item_result = {
                "image": input_image_path,
                "id": data["id"][0],
                "success": True,
            }
            if progress is not None:
                progress.set_description(str(input_image_path))
            brain_masks: list[dict[str, Any]] = []
            if do_brain_mask or self.crop_segmentation_input_to_brain_mask:
                try:
                    brain_masks = self.predict_synthstrip_batch(data)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    print(f"Error during prediction for {input_image_path}:")
                    traceback.print_exc()
                    continue
                if do_brain_mask:
                    for i in range(len(brain_masks)):
                        brain_mask_output_path = (
                            data["brain_mask_output"][i]
                            .removesuffix(".nii")
                            .removesuffix(".nii.gz")
                        )
                        Path(brain_mask_output_path).parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        saver(
                            brain_masks[i]["brain_mask"][None].cpu(),
                            filename=brain_mask_output_path,
                        )
                        item_result["synthstrip"] = (
                            brain_mask_output_path + self.output_extension
                        )

            if do_segmentation:
                try:
                    synthseg_input = data["image"]
                    brain_mask = None
                    if self.crop_segmentation_input_to_brain_mask:
                        assert len(brain_masks) == 1, (
                            "Batch size must be 1 when cropping to brain mask"
                        )
                        brain_mask = brain_masks[0]["brain_mask"]
                    segmentations = self.predict_synthseg_batch(
                        synthseg_input, brain_mask=brain_mask
                    )
                    del synthseg_input
                except KeyboardInterrupt:
                    raise
                except Exception:
                    print(f"Error during prediction for {input_image_path}:")
                    traceback.print_exc()
                    continue
                for i in range(len(segmentations)):
                    output_path = (
                        data["output"][i].removesuffix(".nii").removesuffix(".nii.gz")
                    )
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    saver(
                        segmentations[i]["segmentation"][None].cpu(),
                        filename=output_path,
                    )
                    item_result["synthseg"] = output_path + self.output_extension
                    if do_qc and qc_output_path is not None:
                        self._append_qc_row(
                            qc_output_path=qc_output_path,
                            input_image_path=input_image_path,
                            qc_scores=segmentations[i]["qc_scores"],
                        )
                        item_result["qc"] = segmentations[i]["qc_scores"]
                        item_result["qc_output"] = qc_output_path.as_posix()
            if callback is not None:
                callback(item_result)


def main():
    Fire(Application)


if __name__ == "__main__":
    example_file = Path(__file__).parent / "res" / "spgr_unstrip.nii"
    output_file = Path(__file__).parent / "res" / "spgr_unstrip_segmentation.nii"
    app = Application(parcellation=True)
    app.run(
        input_paths=example_file.as_posix(),
        segmentation_out=output_file.as_posix(),
    )
