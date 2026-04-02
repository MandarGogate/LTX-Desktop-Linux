"""Tests for GGUF model loader discovery and info."""

from __future__ import annotations

from pathlib import Path

from services.gguf_loader.gguf_loader import GGUFModelLoader, get_gguf_filename, get_gguf_repo_id


class TestGGUFFilenameResolution:
    def test_unsloth_q8_0(self) -> None:
        name = get_gguf_filename("Q8_0", source="unsloth")
        assert name is not None
        assert "Q8_0" in name
        assert name.endswith(".gguf")

    def test_unsloth_q4_0(self) -> None:
        name = get_gguf_filename("Q4_0", source="unsloth")
        assert name is not None
        assert "Q4_0" in name

    def test_unknown_quant_returns_none(self) -> None:
        name = get_gguf_filename("Q99_X", source="unsloth")
        assert name is None

    def test_unknown_source_returns_none(self) -> None:
        name = get_gguf_filename("Q8_0", source="nonexistent")
        assert name is None

    def test_repo_id(self) -> None:
        repo_id = get_gguf_repo_id("unsloth")
        assert repo_id == "unsloth/LTX-2.3-GGUF"


class TestGGUFModelDiscovery:
    def test_find_returns_none_when_no_files(self, tmp_path: Path) -> None:
        loader = GGUFModelLoader(tmp_path)
        assert loader.find_gguf_model() is None
        assert loader.is_available() is False

    def test_find_returns_file_in_gguf_subdir(self, tmp_path: Path) -> None:
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        model_file = gguf_dir / "LTX-2.3-Q8_0.gguf"
        model_file.write_bytes(b"\x00" * 100)

        loader = GGUFModelLoader(tmp_path)
        result = loader.find_gguf_model("Q8_0")
        assert result is not None
        assert result.name == "LTX-2.3-Q8_0.gguf"

    def test_find_returns_file_in_models_dir(self, tmp_path: Path) -> None:
        model_file = tmp_path / "LTX-2.3-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00" * 100)

        loader = GGUFModelLoader(tmp_path)
        result = loader.find_gguf_model("Q4_K_M")
        assert result is not None
        assert "Q4_K_M" in result.name

    def test_prefers_requested_quant(self, tmp_path: Path) -> None:
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        (gguf_dir / "LTX-2.3-Q8_0.gguf").write_bytes(b"\x00" * 100)
        (gguf_dir / "LTX-2.3-Q4_0.gguf").write_bytes(b"\x00" * 100)

        loader = GGUFModelLoader(tmp_path)
        result = loader.find_gguf_model("Q4_0")
        assert result is not None
        assert "Q4_0" in result.name

    def test_fallback_to_any_gguf(self, tmp_path: Path) -> None:
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        (gguf_dir / "some-model.gguf").write_bytes(b"\x00" * 100)

        loader = GGUFModelLoader(tmp_path)
        result = loader.find_gguf_model("Q99_X")
        assert result is not None
        assert result.name == "some-model.gguf"

    def test_ignores_zimage_gguf_when_selecting_video_model(self, tmp_path: Path) -> None:
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        (gguf_dir / "z-image-turbo-BF16.gguf").write_bytes(b"\x00" * 100)
        (gguf_dir / "ltx-2.3-22b-dev-Q8_0.gguf").write_bytes(b"\x00" * 100)

        loader = GGUFModelLoader(tmp_path)
        result = loader.find_gguf_model("Q8_0")
        assert result is not None
        assert result.name == "ltx-2.3-22b-dev-Q8_0.gguf"


class TestGGUFInfo:
    def test_info_empty_when_no_files(self, tmp_path: Path) -> None:
        loader = GGUFModelLoader(tmp_path)
        info = loader.get_gguf_info()
        assert info["available_models"] == []

    def test_info_lists_available_models(self, tmp_path: Path) -> None:
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        (gguf_dir / "LTX-2.3-Q8_0.gguf").write_bytes(b"\x00" * 1024)
        (gguf_dir / "LTX-2.3-Q4_0.gguf").write_bytes(b"\x00" * 512)

        loader = GGUFModelLoader(tmp_path)
        info = loader.get_gguf_info()
        models = info["available_models"]
        assert len(models) == 2

        names = {m["filename"] for m in models}
        assert "LTX-2.3-Q8_0.gguf" in names
        assert "LTX-2.3-Q4_0.gguf" in names

    def test_info_detects_quant_level(self, tmp_path: Path) -> None:
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        (gguf_dir / "LTX-2.3-Q4_K_M.gguf").write_bytes(b"\x00" * 100)

        loader = GGUFModelLoader(tmp_path)
        info = loader.get_gguf_info()
        model = info["available_models"][0]
        assert model["quant_level"] == "Q4_K_M"
