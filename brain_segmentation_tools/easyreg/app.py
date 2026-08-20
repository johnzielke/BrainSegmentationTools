from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import nibabel as nib
import nibabel.affines
import numpy as np
import torch
import torch.nn.functional as F

from brain_segmentation_tools.easyreg.model import EasyRegDeformableNet

if TYPE_CHECKING:
    from brain_segmentation_tools.model_manager import ModelManager


def _ensure_sequence(value: str | Path | list[str] | list[Path] | None, *, name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return [str(value)]
    items = [str(item) for item in value]
    if not items:
        raise ValueError(f"{name} must not be empty")
    return items


def _broadcast_optional_list(
    value: str | Path | list[str] | list[Path] | None,
    *,
    name: str,
    expected_length: int,
) -> list[str | None]:
    if value is None:
        return [None] * expected_length
    items = _ensure_sequence(value, name=name)
    assert items is not None
    if len(items) == 1 and expected_length > 1:
        return cast(list[str | None], [items[0]] * expected_length)
    if len(items) != expected_length:
        raise ValueError(f"{name} must have length 1 or {expected_length}, got {len(items)}")
    return cast(list[str | None], items)


def _validate_output_lists(
    *, deformation_field_out: list[str] | None, warped_image_out: list[str] | None, expected_length: int
) -> tuple[list[str | None], list[str | None]]:
    field_items = (
        _broadcast_optional_list(
            deformation_field_out,
            name="deformation_field_out",
            expected_length=expected_length,
        )
        if deformation_field_out is not None
        else [None] * expected_length
    )
    warped_items = (
        _broadcast_optional_list(
            warped_image_out,
            name="warped_image_out",
            expected_length=expected_length,
        )
        if warped_image_out is not None
        else [None] * expected_length
    )
    if all(item is None for item in field_items) and all(item is None for item in warped_items):
        raise ValueError("At least one of deformation_field_out or warped_image_out must be provided")
    return field_items, warped_items


def _load_nifti(path: str) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    image = cast(nib.Nifti1Image, nib.load(path))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image at {path}, got shape {data.shape}")
    return data, np.asarray(image.affine, dtype=np.float64), image.header


def _save_nifti(data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data, affine, header=header.copy())
    nib.save(out, str(path_obj))


def _normalize_intensity(image: np.ndarray) -> np.ndarray:
    positive = image[image > 0]
    values = positive if positive.size else image.reshape(-1)
    min_value = float(np.percentile(values, 0.5))
    max_value = float(np.percentile(values, 99.5))
    image = np.clip(image, min_value, max_value)
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def _compose_affine(
    pre_affine: np.ndarray | None,
    reference_affine: np.ndarray,
    moving_affine: np.ndarray,
) -> np.ndarray:
    if pre_affine is None:
        return np.linalg.inv(moving_affine) @ reference_affine
    return np.linalg.inv(moving_affine) @ pre_affine @ reference_affine


def _images_are_aligned(
    pre_affine: np.ndarray | None,
    reference_affine: np.ndarray,
    moving_affine: np.ndarray,
) -> bool:
    if pre_affine is not None:
        return False
    return np.allclose(reference_affine, moving_affine, atol=1e-5)


def _affine_grid_from_matrix(matrix_vox: np.ndarray, shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    depth, height, width = shape
    z, y, x = torch.meshgrid(
        torch.arange(depth, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    coords = torch.stack([z, y, x, ones], dim=-1)
    transformed = torch.as_tensor(matrix_vox, dtype=torch.float32, device=device) @ coords.reshape(-1, 4).T
    transformed = transformed.T.reshape(depth, height, width, 4)[..., :3]
    return transformed


def _voxel_to_normalized(coords_zyx: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    depth, height, width = spatial_shape
    z = coords_zyx[..., 0]
    y = coords_zyx[..., 1]
    x = coords_zyx[..., 2]
    x_norm = 2.0 * x / max(width - 1, 1) - 1.0
    y_norm = 2.0 * y / max(height - 1, 1) - 1.0
    z_norm = 2.0 * z / max(depth - 1, 1) - 1.0
    return torch.stack([x_norm, y_norm, z_norm], dim=-1)


def _warp_image(image: torch.Tensor, sample_coords_zyx: torch.Tensor) -> torch.Tensor:
    grid = _voxel_to_normalized(sample_coords_zyx, cast(tuple[int, int, int], tuple(image.shape[-3:])))
    warped = F.grid_sample(
        image,
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped


@dataclass
class EasyRegApplication:
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    dev_mode: bool | None = None
    model: EasyRegDeformableNet | None = None
    model_manager: ModelManager | None = None

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)
        if self.model_manager is None:
            from brain_segmentation_tools.model_manager import ModelManager

            self.model_manager = ModelManager(dev_mode=self.dev_mode)

    @property
    def easyreg_model(self) -> EasyRegDeformableNet:
        if self.model is None:
            assert self.model_manager is not None
            model_path = self.model_manager.get_model_path(
                model_name="easyreg",
                model_type="deformable_field",
                version="1.0",
                allow_h5_in_dev=True,
            )
            model = EasyRegDeformableNet()
            model.load_weights(model_path.as_posix())
            model = model.to(torch.device(self.device), dtype=self.dtype)
            model.eval()
            self.model = model
        return self.model

    @torch.inference_mode()
    def predict_deformable_fields(
        self,
        *,
        reference_images: str | Path | list[str] | list[Path],
        moving_images: list[str] | list[Path] | str | Path,
        pre_affines: list[np.ndarray | None] | np.ndarray | None = None,
        deformation_field_out: str | Path | list[str] | list[Path] | None = None,
        warped_image_out: str | Path | list[str] | list[Path] | None = None,
    ) -> list[dict[str, str | None]]:
        moving_items = _ensure_sequence(moving_images, name="moving_images")
        assert moving_items is not None
        if isinstance(reference_images, (str, Path)):
            reference_items = [str(reference_images)] * len(moving_items)
        else:
            reference_items = [str(item) for item in reference_images]
            if len(reference_items) == 1 and len(moving_items) > 1:
                reference_items = reference_items * len(moving_items)
            elif len(reference_items) != len(moving_items):
                raise ValueError("reference_images must have length 1 or match moving_images")

        if pre_affines is None:
            affine_items: list[np.ndarray | None] = [None] * len(moving_items)
        elif isinstance(pre_affines, np.ndarray):
            affine_items = [pre_affines] * len(moving_items)
        else:
            affine_items = list(pre_affines)
            if len(affine_items) == 1 and len(moving_items) > 1:
                affine_items = affine_items * len(moving_items)
            elif len(affine_items) != len(moving_items):
                raise ValueError("pre_affines must have length 1 or match moving_images")

        field_out_items, warped_out_items = _validate_output_lists(
            deformation_field_out=_ensure_sequence(deformation_field_out, name="deformation_field_out"),
            warped_image_out=_ensure_sequence(warped_image_out, name="warped_image_out"),
            expected_length=len(moving_items),
        )

        outputs: list[dict[str, str | None]] = []
        for reference_path, moving_path, pre_affine, field_path, warped_path in zip(
            reference_items, moving_items, affine_items, field_out_items, warped_out_items, strict=True
        ):
            ref_data, ref_affine, ref_header = _load_nifti(reference_path)
            mov_data, mov_affine, mov_header = _load_nifti(moving_path)

            ref_tensor = torch.from_numpy(_normalize_intensity(ref_data)).unsqueeze(0).unsqueeze(0)
            mov_tensor = torch.from_numpy(_normalize_intensity(mov_data)).unsqueeze(0).unsqueeze(0)
            ref_tensor = ref_tensor.to(self.device, dtype=self.dtype)
            mov_tensor = mov_tensor.to(self.device, dtype=self.dtype)

            if _images_are_aligned(pre_affine, ref_affine, mov_affine):
                affine_grid = _affine_grid_from_matrix(np.eye(4, dtype=np.float64), ref_data.shape, ref_tensor.device)
                model_moving = mov_tensor
            else:
                affine_vox = _compose_affine(pre_affine, ref_affine, mov_affine)
                affine_grid = _affine_grid_from_matrix(affine_vox, ref_data.shape, ref_tensor.device)
                model_moving = _warp_image(mov_tensor, affine_grid)
            flow = self.easyreg_model(ref_tensor, model_moving)
            total_grid = affine_grid + flow.squeeze(0)

            if field_path is not None:
                ras_coords = nibabel.affines.apply_affine(
                    mov_affine,
                    total_grid.detach().cpu().numpy()[..., ::-1],
                )
                _save_nifti(ras_coords.astype(np.float32), ref_affine, ref_header, field_path)

            if warped_path is not None:
                warped = _warp_image(mov_tensor, total_grid).squeeze().detach().cpu().numpy().astype(np.float32)
                _save_nifti(warped, ref_affine, mov_header, warped_path)

            outputs.append(
                {
                    "reference_image": reference_path,
                    "moving_image": moving_path,
                    "deformation_field_out": field_path,
                    "warped_image_out": warped_path,
                }
            )
        return outputs


def predict_deformable_fields(
    *,
    reference_images: str | Path | list[str] | list[Path],
    moving_images: list[str] | list[Path] | str | Path,
    pre_affines: list[np.ndarray | None] | np.ndarray | None = None,
    deformation_field_out: str | Path | list[str] | list[Path] | None = None,
    warped_image_out: str | Path | list[str] | list[Path] | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    dev_mode: bool | None = None,
) -> list[dict[str, str | None]]:
    app = EasyRegApplication(device=device, dtype=dtype, dev_mode=dev_mode)
    return app.predict_deformable_fields(
        reference_images=reference_images,
        moving_images=moving_images,
        pre_affines=pre_affines,
        deformation_field_out=deformation_field_out,
        warped_image_out=warped_image_out,
    )
