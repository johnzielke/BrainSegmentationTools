from __future__ import annotations

from pathlib import Path

import pytest

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
