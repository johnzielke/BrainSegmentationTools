from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brain_segmentation_tools.model_manager import ModelManager


def test_model_manager_uses_env_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_cache_dir = tmp_path / "custom-model-cache"
    monkeypatch.setenv(
        ModelManager.MODEL_CACHE_DIR_ENV_VAR, custom_cache_dir.as_posix()
    )
    manager = ModelManager(dev_mode=False)
    assert manager.cache_dir == custom_cache_dir
    assert manager.cache_dir.exists()


def test_model_download_and_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_name = "synthstrip"
    model_type = "normal"
    version = "1"

    manager = ModelManager(dev_mode=False)
    manager.cache_dir = tmp_path / "models"
    manager.cache_dir.mkdir(parents=True, exist_ok=True)

    download_calls: list[tuple[str, Path]] = []
    original_download = ModelManager._download_pt

    def wrapped_download(
        url: str, target_path: Path, *, expected_hash: str | None = None
    ) -> None:
        download_calls.append((url, Path(target_path)))
        original_download(url, target_path, expected_hash=expected_hash)

    monkeypatch.setattr(ModelManager, "_download_pt", staticmethod(wrapped_download))

    spec = manager.get_spec(
        model_name=model_name, model_type=model_type, version=version
    )

    try:
        first_path = manager.get_model_path(
            model_name=model_name,
            model_type=model_type,
            version=version,
            allow_h5_in_dev=False,
        )
    except RuntimeError as exc:
        if "Failed downloading model" in str(exc):
            pytest.skip(
                "Network unavailable or model host unreachable; "
                "skipping download/cache integration test"
            )
        raise

    assert first_path.exists()
    assert len(download_calls) == 1
    assert manager._file_hash(first_path) == spec.content_hash

    second_path = manager.get_model_path(
        model_name=model_name,
        model_type=model_type,
        version=version,
        allow_h5_in_dev=False,
    )

    assert second_path == first_path
    assert second_path.exists()
    assert len(download_calls) == 1

    first_path.write_bytes(b"corrupted-cache")
    assert manager._file_hash(first_path) != spec.content_hash

    repaired_path = manager.get_model_path(
        model_name=model_name,
        model_type=model_type,
        version=version,
        allow_h5_in_dev=False,
    )

    assert repaired_path == first_path
    assert repaired_path.exists()
    assert len(download_calls) == 2
    assert manager._file_hash(repaired_path) == spec.content_hash


def test_get_model_state_dict_extracts_reference_synthstrip_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ModelManager(dev_mode=False)
    reference_checkpoint = tmp_path / "synthstrip_reference.pt"
    expected_state_dict = {
        "encoder.0.0.conv.weight": torch.ones((1, 1, 3, 3, 3)),
        "encoder.0.0.conv.bias": torch.zeros(1),
    }
    torch.save(
        {
            "epoch": 12,
            "model_state_dict": expected_state_dict,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
        },
        reference_checkpoint,
    )

    def fake_get_model_path(**_: object) -> Path:
        return reference_checkpoint

    monkeypatch.setattr(manager, "get_model_path", fake_get_model_path)

    state_dict = manager.get_model_state_dict(
        model_name="synthstrip",
        model_type="normal",
        version="1",
    )

    assert set(state_dict) == set(expected_state_dict)
    for key, tensor in expected_state_dict.items():
        assert torch.equal(state_dict[key], tensor)


def test_get_model_state_dict_uses_model_path_not_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ModelManager(dev_mode=False)
    checkpoint_path = tmp_path / "wrapped_checkpoint.pt"
    expected_state_dict = {
        "weight": torch.ones((1, 1, 3, 3, 3)),
    }
    torch.save({"state_dict": expected_state_dict}, checkpoint_path)

    def fake_get_model_path(**_: object) -> Path:
        return checkpoint_path

    def fail_to_pt(*_: object, **__: object) -> Path:
        raise AssertionError("get_model_state_dict should not call _to_pt")

    monkeypatch.setattr(manager, "get_model_path", fake_get_model_path)
    monkeypatch.setattr(manager, "_to_pt", fail_to_pt)

    state_dict = manager.get_model_state_dict(
        model_name="synthstrip",
        model_type="normal",
        version="1",
    )

    assert set(state_dict) == set(expected_state_dict)
    assert torch.equal(state_dict["weight"], expected_state_dict["weight"])


def test_to_pt_wraps_synthstrip_checkpoint_with_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ModelManager(dev_mode=False)
    manager.cache_dir = tmp_path / "models"
    manager.cache_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint = tmp_path / "synthstrip_reference.pt"
    expected_state_dict = {
        "encoder.0.0.conv.weight": torch.ones((1, 1, 3, 3, 3)),
        "encoder.0.0.conv.bias": torch.zeros(1),
    }
    torch.save(
        {
            "epoch": 7,
            "model_state_dict": expected_state_dict,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
        },
        source_checkpoint,
    )

    spec = manager.get_spec(model_name="synthstrip", model_type="normal", version="1")

    def fake_get_model_path(**_: object) -> Path:
        return source_checkpoint

    monkeypatch.setattr(manager, "get_model_path", fake_get_model_path)

    converted_path = manager._to_pt(spec)
    checkpoint = torch.load(converted_path, map_location="cpu", weights_only=False)

    assert checkpoint["metadata"] == {
        "model_name": "synthstrip",
        "model_type": "normal",
        "version": "1",
        "framework": "torch_state_dict",
    }
    assert "optimizer_state_dict" not in checkpoint
    assert "epoch" not in checkpoint
    assert set(checkpoint["state_dict"]) == set(expected_state_dict)
    for key, tensor in expected_state_dict.items():
        assert torch.equal(checkpoint["state_dict"][key], tensor)


@pytest.mark.parametrize(
    ("model_type", "expected_h5"),
    [
        ("segmentation_robust", "mri_synthseg/synthseg_robust_2.0.h5"),
        ("qc", "mri_synthseg/synthseg_qc_2.0.h5"),
    ],
)
def test_model_manager_has_expected_synthseg_v2_h5_specs(
    model_type: str, expected_h5: str
) -> None:
    spec = ModelManager(dev_mode=True).get_spec(
        model_name="synthseg",
        model_type=model_type,
        version="2.0",
    )
    assert spec.freesurfer_h5 == expected_h5
