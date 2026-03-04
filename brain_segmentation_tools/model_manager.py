from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from platformdirs import user_cache_dir

from brain_segmentation_tools.unet_pytorch import UNet


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_type: str
    version: str
    framework: str
    pt_filename: str
    download_url: str
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
    def __init__(self, *, dev_mode: bool | None = None):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.freesurfer_root = self.repo_root / "dev" / "freesurfer"
        cache_root = Path(
            user_cache_dir("brain_segmentation_tools", "brain_segmentation_tools")
        )
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
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthseg_segmentation_1.0.6d8944232ceb.pt",
                freesurfer_h5="mri_synthseg/synthseg_1.0.h5",
                tf_prefix="unet",
                labels_model="segmentation",
                content_hash="6d8944232ceb",
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
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthseg_segmentation_2.0.6271d6360574.pt",
                freesurfer_h5="mri_synthseg/synthseg_2.0.h5",
                tf_prefix="unet",
                labels_model="segmentation",
                content_hash="6271d6360574",
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
                model_type="parcellation",
                version="2.0",
                framework="synthseg_unet",
                pt_filename="synthseg_parcellation_2.0.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthseg_parcellation_2.0.1945f2f4e32b.pt",
                freesurfer_h5="mri_synthseg/synthseg_parc_2.0.h5",
                tf_prefix="unet_parc",
                labels_model="parcellation",
                content_hash="1945f2f4e32b",
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
                model_name="synthstrip",
                model_type="normal",
                version="1",
                framework="torch_state_dict",
                pt_filename="synthstrip_normal_1.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthstrip_normal_1.37417f802196.pt",
                freesurfer_pt="mri_synthstrip/synthstrip.1.pt",
                content_hash="37417f802196",
            ),
            ModelSpec(
                model_name="synthstrip",
                model_type="nocsf",
                version="1",
                framework="torch_state_dict",
                pt_filename="synthstrip_nocsf_1.pt",
                download_url="https://github.com/johnzielke/BrainSegmentationToolsData/releases/download/models-v1/synthstrip_nocsf_1.62bf01137c45.pt",
                freesurfer_pt="mri_synthstrip/synthstrip.nocsf.1.pt",
                content_hash="62bf01137c45",
            ),
        ]

    @property
    def configured_models(self) -> list[ModelSpec]:
        return list(self._specs)

    def get_spec(self, *, model_name: str, model_type: str, version: str) -> ModelSpec:
        clean_version = str(version).removeprefix("v")
        for spec in self._specs:
            if (
                spec.model_name == model_name
                and spec.model_type == model_type
                and spec.version == clean_version
            ):
                return spec
        raise KeyError(
            f"Model not configured: {model_name}:{model_type}:{clean_version}"
        )

    def get_model_path(
        self,
        *,
        model_name: str,
        model_type: str,
        version: str,
        allow_h5_in_dev: bool = True,
    ) -> Path:
        spec = self.get_spec(
            model_name=model_name, model_type=model_type, version=version
        )

        if self.dev_mode:
            dev_pt = self._dev_pt_path(spec)
            if dev_pt is not None and dev_pt.exists():
                return dev_pt
            dev_h5 = self._dev_h5_path(spec)
            if allow_h5_in_dev and dev_h5 is not None and dev_h5.exists():
                return dev_h5

        return self._cached_pt_path(spec)

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
        spec = self.get_spec(
            model_name=model_name, model_type=model_type, version=version
        )
        if spec.framework != "synthseg_unet":
            raise ValueError(f"{spec.key} is not an h5-backed UNet model")
        h5_path = self._dev_h5_path(spec)
        if h5_path is None or not h5_path.exists():
            raise FileNotFoundError(f"Missing source h5 for {spec.key}: {h5_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._convert_unet_h5_to_pt(
            spec=spec, h5_path=h5_path, output_pt_path=output_path
        )
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
        self._download_pt(spec.download_url, pt_path, expected_hash=spec.content_hash)
        return pt_path

    @staticmethod
    def _file_hash(path: Path) -> str:
        return sha256(path.read_bytes()).hexdigest()[:12]

    @staticmethod
    def _download_pt(
        url: str, target_path: Path, *, expected_hash: str | None = None
    ) -> None:
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
            shutil.copy2(source_pt, temp_out)
            temp_out.chmod(0o644)
            return temp_out

        dev_h5 = self._dev_h5_path(spec)
        temp_out = self.cache_dir / "converted" / spec.pt_filename
        temp_out.parent.mkdir(parents=True, exist_ok=True)
        if temp_out.exists():
            temp_out.unlink()
        if dev_h5 is not None and dev_h5.exists():
            self._convert_unet_h5_to_pt(
                spec=spec, h5_path=dev_h5, output_pt_path=temp_out
            )
            return temp_out

        return self._cached_pt_path(spec)

    @staticmethod
    def _convert_unet_h5_to_pt(
        *, spec: ModelSpec, h5_path: Path, output_pt_path: Path
    ) -> None:
        if spec.unet_kwargs is None:
            raise ValueError(f"Missing unet_kwargs in model config for {spec.key}")
        model = UNet(
            **spec.unet_kwargs,
        )
        model.load_from_tensorflow(h5_path.as_posix(), prefix=spec.tf_prefix)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": {
                    "source": h5_path.as_posix(),
                    "model_name": spec.model_name,
                    "model_type": spec.model_type,
                    "version": spec.version,
                    "framework": spec.framework,
                },
            },
            output_pt_path,
        )
