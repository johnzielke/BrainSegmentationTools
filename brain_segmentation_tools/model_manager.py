from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
from platformdirs import user_cache_dir

from brain_segmentation_tools.easyreg.model import EasyRegDeformableNet
from brain_segmentation_tools.qc_model import QCSynthSegRegressor
from brain_segmentation_tools.unet_pytorch import UNet


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_type: str
    version: str
    framework: str
    pt_filename: str
    download_url: str | None
    freesurfer_h5: str | None = None
    freesurfer_pt: str | None = None
    tf_prefix: str | None = None
    labels_model: str | None = None
    content_hash: str | None = None
    unet_kwargs: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        return f"{self.model_name}:{self.model_type}:{self.version}"


class ModelManager:
    MODEL_CACHE_DIR_ENV_VAR = "BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR"

    def __init__(self, *, dev_mode: bool | None = None):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.freesurfer_root = self.repo_root / "dev" / "freesurfer"
        cache_dir_from_env = os.environ.get(self.MODEL_CACHE_DIR_ENV_VAR)
        if cache_dir_from_env:
            self.cache_dir = Path(cache_dir_from_env).expanduser()
        else:
            cache_root = Path(user_cache_dir("brain_segmentation_tools", "brain_segmentation_tools"))
            self.cache_dir = cache_root / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dev_mode = self.freesurfer_root.exists() if dev_mode is None else dev_mode
        self._specs = self._build_specs()

    @staticmethod
    def _build_specs() -> list[ModelSpec]:
        return [
            ModelSpec(
                model_name="synthseg",
                model_type="segmentation",
                version="1.0",
                framework="synthseg_unet",
                pt_filename="synthseg_segmentation_1.0.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthseg_segmentation_1.0.dc76d54e9f10.pt",
                freesurfer_h5="mri_synthseg/synthseg_1.0.h5",
                tf_prefix="unet",
                labels_model="segmentation",
                content_hash="dc76d54e9f10",
                unet_kwargs={
                    "nb_features": 24,
                    "in_channels": 1,
                    "nb_levels": 5,
                    "conv_size": 3,
                    "nb_labels": 32,
                    "feat_mult": 2,
                    "activation": "elu",
                    "nb_conv_per_level": 2,
                    "batch_norm": True,
                },
            ),
            ModelSpec(
                model_name="synthseg",
                model_type="segmentation",
                version="2.0",
                framework="synthseg_unet",
                pt_filename="synthseg_segmentation_2.0.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthseg_segmentation_2.0.823fad05ee75.pt",
                freesurfer_h5="mri_synthseg/synthseg_2.0.h5",
                tf_prefix="unet",
                labels_model="segmentation",
                content_hash="823fad05ee75",
                unet_kwargs={
                    "nb_features": 24,
                    "in_channels": 1,
                    "nb_levels": 5,
                    "conv_size": 3,
                    "nb_labels": 33,
                    "feat_mult": 2,
                    "activation": "elu",
                    "nb_conv_per_level": 2,
                    "batch_norm": True,
                },
            ),
            ModelSpec(
                model_name="synthseg",
                model_type="segmentation_robust",
                version="2.0",
                framework="synthseg_robust_unet",
                pt_filename="synthseg_segmentation_robust_2.0.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthseg_segmentation_robust_2.0.bc037d43812d.pt",
                freesurfer_h5="mri_synthseg/synthseg_robust_2.0.h5",
                labels_model="segmentation",
                content_hash="bc037d43812d",
            ),
            ModelSpec(
                model_name="synthseg",
                model_type="parcellation",
                version="2.0",
                framework="synthseg_unet",
                pt_filename="synthseg_parcellation_2.0.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthseg_parcellation_2.0.a8822f886c5c.pt",
                freesurfer_h5="mri_synthseg/synthseg_parc_2.0.h5",
                tf_prefix="unet_parc",
                labels_model="parcellation",
                content_hash="a8822f886c5c",
                unet_kwargs={
                    "nb_features": 24,
                    "in_channels": 3,
                    "nb_levels": 5,
                    "conv_size": 3,
                    "nb_labels": 69,
                    "feat_mult": 2,
                    "activation": "elu",
                    "nb_conv_per_level": 2,
                    "batch_norm": True,
                },
            ),
            ModelSpec(
                model_name="synthseg",
                model_type="qc",
                version="2.0",
                framework="synthseg_qc_regressor",
                pt_filename="synthseg_qc_2.0.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthseg_qc_2.0.ce460ebf1479.pt",
                freesurfer_h5="mri_synthseg/synthseg_qc_2.0.h5",
                labels_model="qc",
                content_hash="ce460ebf1479",
            ),
            ModelSpec(
                model_name="easyreg",
                model_type="deformable_field",
                version="1.0",
                framework="easyreg_deformable",
                pt_filename="easyreg_deformable_field_1.0.pt",
                download_url=None,
                freesurfer_h5="mri_easyreg/easyreg_v10_230103.h5",
                content_hash=None,
            ),
            ModelSpec(
                model_name="synthstrip",
                model_type="normal",
                version="1",
                framework="torch_state_dict",
                pt_filename="synthstrip_normal_1.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthstrip_normal_1.7aa3f5db738c.pt",
                freesurfer_pt="mri_synthstrip/synthstrip.1.pt",
                content_hash="7aa3f5db738c",
            ),
            ModelSpec(
                model_name="synthstrip",
                model_type="nocsf",
                version="1",
                framework="torch_state_dict",
                pt_filename="synthstrip_nocsf_1.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthstrip_nocsf_1.805a73fdceb1.pt",
                freesurfer_pt="mri_synthstrip/synthstrip.nocsf.1.pt",
                content_hash="805a73fdceb1",
            ),
            ModelSpec(
                model_name="contrast_classifier",
                model_type="normal",
                version="1",
                framework="torch_state_dict",
                pt_filename="contrast_classifier_1.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/contrast_classifier_1.1d5e14b2362e.pt",
                freesurfer_pt="mri_contrast_classifier/contrast_classifier.1.pt",
                content_hash="1d5e14b2362e",
            ),
        ]

    @property
    def configured_models(self) -> list[ModelSpec]:
        return list(self._specs)

    def get_spec(self, *, model_name: str, model_type: str, version: str) -> ModelSpec:
        clean_version = str(version).removeprefix("v")
        for spec in self._specs:
            if spec.model_name == model_name and spec.model_type == model_type and spec.version == clean_version:
                return spec
        raise KeyError(f"Model not configured: {model_name}:{model_type}:{clean_version}")

    def get_model_path(
        self,
        *,
        model_name: str,
        model_type: str,
        version: str,
        allow_h5_in_dev: bool = True,
    ) -> Path:
        spec = self.get_spec(model_name=model_name, model_type=model_type, version=version)

        if self.dev_mode:
            dev_pt = self._dev_pt_path(spec)
            if dev_pt is not None and dev_pt.exists():
                return dev_pt
            dev_h5 = self._dev_h5_path(spec)
            if allow_h5_in_dev and dev_h5 is not None and dev_h5.exists():
                return dev_h5

        return self._cached_pt_path(spec)

    def get_model_state_dict(
        self,
        *,
        model_name: str,
        model_type: str,
        version: str,
    ) -> Any:
        model_path = self.get_model_path(
            model_name=model_name,
            model_type=model_type,
            version=version,
            allow_h5_in_dev=False,
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        return self._extract_state_dict(checkpoint)

    def save_all_converted(self, output_dir: str | Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Path] = {}
        for spec in self._specs:
            pt_path = self._to_pt(spec)
            digest = self._file_hash(pt_path)
            out_name = f"{Path(spec.pt_filename).stem}.{digest}.pt"
            out_path = output_dir / out_name
            if out_path.exists():
                out_path.unlink()
            shutil.copy2(pt_path, out_path)
            out_path.chmod(0o644)
            results[spec.key] = out_path
        return results

    def convert_h5_to_pt(
        self,
        *,
        model_name: str,
        model_type: str,
        version: str,
        output_path: str | Path,
    ) -> Path:
        spec = self.get_spec(model_name=model_name, model_type=model_type, version=version)
        if spec.framework not in {
            "synthseg_unet",
            "synthseg_robust_unet",
            "synthseg_qc_regressor",
            "easyreg_deformable",
        }:
            raise ValueError(f"{spec.key} is not an h5-backed model")
        h5_path = self._dev_h5_path(spec)
        if h5_path is None or not h5_path.exists():
            raise FileNotFoundError(f"Missing source h5 for {spec.key}: {h5_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if spec.framework == "synthseg_robust_unet":
            self._convert_robust_h5_to_pt(spec=spec, h5_path=h5_path, output_pt_path=output_path)
        elif spec.framework == "synthseg_qc_regressor":
            self._convert_qc_h5_to_pt(spec=spec, h5_path=h5_path, output_pt_path=output_path)
        elif spec.framework == "easyreg_deformable":
            self._convert_easyreg_h5_to_pt(spec=spec, h5_path=h5_path, output_pt_path=output_path)
        else:
            self._convert_unet_h5_to_pt(spec=spec, h5_path=h5_path, output_pt_path=output_path)
        return output_path

    def _dev_h5_path(self, spec: ModelSpec) -> Path | None:
        if spec.freesurfer_h5 is None:
            return None
        return self.freesurfer_root / spec.freesurfer_h5

    def _dev_pt_path(self, spec: ModelSpec) -> Path | None:
        if spec.freesurfer_pt is not None:
            return self.freesurfer_root / spec.freesurfer_pt
        return None

    def _cached_pt_path(self, spec: ModelSpec) -> Path:
        pt_path = self.cache_dir / spec.pt_filename
        if pt_path.exists():
            if spec.content_hash is None:
                return pt_path
            current_hash = self._file_hash(pt_path)
            if current_hash == spec.content_hash:
                return pt_path
            pt_path.unlink()
        if spec.download_url is None:
            raise RuntimeError(
                f"No downloadable artifact configured for {spec.key}; "
                "run in dev_mode with FreeSurfer models available or convert locally."
            )
        self._download_pt(spec.download_url, pt_path, expected_hash=spec.content_hash)
        return pt_path

    @staticmethod
    def _file_hash(path: Path) -> str:
        return sha256(path.read_bytes()).hexdigest()[:12]

    @staticmethod
    def _download_pt(url: str, target_path: Path, *, expected_hash: str | None = None) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            if expected_hash is not None:
                downloaded_hash = ModelManager._file_hash(tmp_path)
                if downloaded_hash != expected_hash:
                    raise RuntimeError(
                        f"Downloaded file hash mismatch for {target_path.name}: "
                        f"expected {expected_hash}, got {downloaded_hash}"
                    )
            tmp_path.replace(target_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed downloading model from {url}") from e

    def _to_pt(self, spec: ModelSpec) -> Path:
        if spec.framework == "torch_state_dict":
            source_pt = self.get_model_path(
                model_name=spec.model_name,
                model_type=spec.model_type,
                version=spec.version,
                allow_h5_in_dev=False,
            )
            temp_out = self.cache_dir / "converted" / spec.pt_filename
            temp_out.parent.mkdir(parents=True, exist_ok=True)
            if temp_out.exists():
                temp_out.unlink()
            checkpoint = torch.load(source_pt, map_location="cpu", weights_only=False)
            torch.save(
                {
                    "state_dict": self._extract_state_dict(checkpoint),
                    "metadata": self._checkpoint_metadata(spec),
                },
                temp_out,
            )
            temp_out.chmod(0o644)
            return temp_out

        dev_h5 = self._dev_h5_path(spec)
        temp_out = self.cache_dir / "converted" / spec.pt_filename
        temp_out.parent.mkdir(parents=True, exist_ok=True)
        if temp_out.exists():
            temp_out.unlink()
        if dev_h5 is not None and dev_h5.exists():
            if spec.framework == "synthseg_robust_unet":
                self._convert_robust_h5_to_pt(spec=spec, h5_path=dev_h5, output_pt_path=temp_out)
            elif spec.framework == "synthseg_qc_regressor":
                self._convert_qc_h5_to_pt(spec=spec, h5_path=dev_h5, output_pt_path=temp_out)
            elif spec.framework == "easyreg_deformable":
                self._convert_easyreg_h5_to_pt(spec=spec, h5_path=dev_h5, output_pt_path=temp_out)
            else:
                self._convert_unet_h5_to_pt(spec=spec, h5_path=dev_h5, output_pt_path=temp_out)
            return temp_out

        return self._cached_pt_path(spec)

    @staticmethod
    def _convert_unet_h5_to_pt(*, spec: ModelSpec, h5_path: Path, output_pt_path: Path) -> None:
        if spec.unet_kwargs is None:
            raise ValueError(f"Missing unet_kwargs in model config for {spec.key}")
        model = UNet(
            **spec.unet_kwargs,
        )
        model.load_from_tensorflow(h5_path.as_posix(), prefix=spec.tf_prefix)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": ModelManager._checkpoint_metadata(spec),
            },
            output_pt_path,
        )

    @staticmethod
    def _convert_robust_h5_to_pt(*, spec: ModelSpec, h5_path: Path, output_pt_path: Path) -> None:
        stage1 = UNet(
            nb_features=24,
            in_channels=1,
            nb_levels=5,
            conv_size=3,
            nb_labels=5,
            feat_mult=2,
            activation="elu",
            nb_conv_per_level=2,
            batch_norm=True,
        )
        stage1.load_from_tensorflow(h5_path.as_posix(), prefix="unet")

        denoiser = UNet(
            nb_features=16,
            in_channels=5,
            nb_levels=5,
            conv_size=5,
            nb_labels=5,
            feat_mult=2,
            activation="elu",
            nb_conv_per_level=2,
            skip_n_concatenations=2,
            batch_norm=True,
        )
        denoiser.load_from_tensorflow(h5_path.as_posix(), prefix="l2l")

        stage2 = UNet(
            nb_features=24,
            in_channels=6,
            nb_levels=5,
            conv_size=3,
            nb_labels=33,
            feat_mult=2,
            activation="elu",
            nb_conv_per_level=2,
            batch_norm=True,
        )
        stage2.load_from_tensorflow(h5_path.as_posix(), prefix="unet2")

        torch.save(
            {
                "state_dict": {
                    "segmentation_model_stage1": stage1.state_dict(),
                    "segmentation_model_denoiser": denoiser.state_dict(),
                    "segmentation_model_stage2": stage2.state_dict(),
                },
                "metadata": ModelManager._checkpoint_metadata(spec),
            },
            output_pt_path,
        )

    @staticmethod
    def _load_labels(model_type: str, version: str) -> list[int]:
        module_version = f"v{str(version).replace('.', '_')}"
        module = import_module(f"brain_segmentation_tools.constants.{module_version}.{model_type}")
        return list(module.RESOURCE["labels"])

    @staticmethod
    def _convert_qc_h5_to_pt(*, spec: ModelSpec, h5_path: Path, output_pt_path: Path) -> None:
        labels_segmentation = np.asarray(ModelManager._load_labels("segmentation", spec.version), dtype=np.int32)
        labels_qc = np.asarray(ModelManager._load_labels("qc", spec.version), dtype=np.int32)
        labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)
        labels_qc = labels_qc[unique_idx]
        model = QCSynthSegRegressor(
            labels_segmentation=labels_segmentation.tolist(),
            labels_qc=labels_qc.tolist(),
        )
        model.load_from_tensorflow(h5_path.as_posix(), prefix="qc")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": ModelManager._checkpoint_metadata(spec),
            },
            output_pt_path,
        )

    @staticmethod
    def _convert_easyreg_h5_to_pt(*, spec: ModelSpec, h5_path: Path, output_pt_path: Path) -> None:
        model = EasyRegDeformableNet()
        model.load_from_tensorflow(h5_path.as_posix())
        torch.save(
            {
                "state_dict": {"easyreg_model": model.state_dict()},
                "metadata": {
                    **ModelManager._checkpoint_metadata(spec),
                    "source_h5": h5_path.as_posix(),
                    "conversion_status": "loaded_from_tensorflow_h5",
                },
            },
            output_pt_path,
        )

    @staticmethod
    def _checkpoint_metadata(spec: ModelSpec) -> dict[str, str]:
        return {
            "model_name": spec.model_name,
            "model_type": spec.model_type,
            "version": spec.version,
            "framework": spec.framework,
        }

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Any:
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint to be a dict or state_dict mapping, got {type(checkpoint).__name__}")

        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]

        if ModelManager._looks_like_state_dict(checkpoint):
            return checkpoint

        raise KeyError(
            "Checkpoint does not contain a recognized state dict. "
            "Expected 'state_dict', 'model_state_dict', or a raw state dict mapping."
        )

    @staticmethod
    def _looks_like_state_dict(candidate: Any) -> bool:
        if not isinstance(candidate, dict) or not candidate:
            return False

        return all(
            isinstance(value, torch.Tensor) or ModelManager._looks_like_nested_state_dict(value)
            for value in candidate.values()
        )

    @staticmethod
    def _looks_like_nested_state_dict(candidate: Any) -> bool:
        if not isinstance(candidate, dict) or not candidate:
            return False

        return all(isinstance(value, torch.Tensor) for value in candidate.values())
