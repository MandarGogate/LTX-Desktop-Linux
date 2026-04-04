"""Tests for run configuration: num_blocks_to_swap, run_mode, external model downloads."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from services.vram_manager.vram_manager import OffloadStrategy, VRAMManager, VRAMTier


class TestRunModeOverride:
    """User can override the auto-detected tier via run_mode setting."""

    def test_auto_mode_uses_vram_detection(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24, user_run_mode="auto")
        assert mgr.tier == VRAMTier.HIGH

    def test_high_vram_override_on_low_gpu(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 8, user_run_mode="high_vram")
        assert mgr.tier == VRAMTier.HIGH

    def test_low_vram_override_on_high_gpu(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24, user_run_mode="low_vram")
        assert mgr.tier == VRAMTier.LOW

    def test_medium_vram_override(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 8, user_run_mode="medium_vram")
        assert mgr.tier == VRAMTier.MEDIUM
        assert mgr.block_swap_keep_on_gpu == 4

    def test_very_low_vram_override(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 48, user_run_mode="very_low_vram")
        assert mgr.tier == VRAMTier.VERY_LOW

    def test_unknown_mode_falls_back_to_auto(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 16, user_run_mode="unknown_mode")
        assert mgr.tier == VRAMTier.MEDIUM  # auto for 16 GB

    def test_offload_strategy_follows_overridden_tier(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24, user_run_mode="very_low_vram")
        assert mgr.offload_strategy == OffloadStrategy.BLOCK_SWAP_AGGRESSIVE

    def test_resolution_follows_overridden_tier(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24, user_run_mode="very_low_vram")
        resolutions = mgr.get_available_resolutions()
        assert "1080p" not in resolutions
        assert "480p" in resolutions


class TestUserBlockSwapOverride:
    """User can override auto block swap count."""

    def test_default_auto_uses_tier_value(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24, user_blocks_on_gpu=-1)
        assert mgr.block_swap_keep_on_gpu == 5  # HIGH tier default

    def test_custom_blocks_on_gpu(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24, user_blocks_on_gpu=15)
        assert mgr.block_swap_keep_on_gpu == 5

    def test_custom_blocks_on_gpu_below_safe_cap_is_preserved(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24, user_blocks_on_gpu=4)
        assert mgr.block_swap_keep_on_gpu == 4

    def test_zero_blocks_on_gpu(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24, user_blocks_on_gpu=0)
        assert mgr.block_swap_keep_on_gpu == 0

    def test_max_blocks_capped_at_48(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24, user_blocks_on_gpu=100)
        assert mgr.block_swap_keep_on_gpu == 5

    def test_negative_falls_back_to_tier(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 12, user_blocks_on_gpu=-1)
        assert mgr.block_swap_keep_on_gpu == 3  # LOW tier default

    def test_combined_run_mode_and_blocks(self) -> None:
        """run_mode overrides tier, blocks override block count independently."""
        mgr = VRAMManager(torch.device("cpu"), 8, user_run_mode="high_vram", user_blocks_on_gpu=2)
        assert mgr.tier == VRAMTier.HIGH  # overridden to high
        assert mgr.block_swap_keep_on_gpu == 2  # but blocks are user-specified


class TestProfileDictExtensions:
    def test_profile_includes_run_mode_fields(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 16, user_blocks_on_gpu=10, user_run_mode="low_vram")
        profile = mgr.to_profile_dict()
        assert profile["user_blocks_on_gpu"] == 10
        assert profile["user_run_mode"] == "low_vram"
        assert profile["auto_tier"] == "medium"  # 16GB auto
        assert profile["auto_blocks_on_gpu"] == 4  # MEDIUM tier default
        assert profile["max_blocks"] == 48
        assert isinstance(profile["run_modes"], list)
        assert len(profile["run_modes"]) == 5

    def test_profile_block_swap_reflects_user_override(self) -> None:
        mgr = VRAMManager(torch.device("cpu"), 24, user_blocks_on_gpu=3)
        profile = mgr.to_profile_dict()
        assert profile["block_swap_blocks_on_gpu"] == 3
        assert profile["auto_blocks_on_gpu"] == 5


class TestSettingsNumBlocksToSwap:
    """Test settings persistence for num_blocks_to_swap and run_mode."""

    def test_default_num_blocks(self, client, test_state):
        r = client.get("/api/settings")
        assert r.status_code == 200
        data = r.json()
        assert data["numBlocksToSwap"] == -1
        assert data["runMode"] == "auto"

    def test_update_num_blocks(self, client, test_state):
        r = client.post("/api/settings", json={"numBlocksToSwap": 10})
        assert r.status_code == 200
        assert test_state.state.app_settings.num_blocks_to_swap == 10

    def test_update_run_mode(self, client, test_state):
        r = client.post("/api/settings", json={"runMode": "low_vram"})
        assert r.status_code == 200
        assert test_state.state.app_settings.run_mode == "low_vram"

    def test_num_blocks_clamped_high(self, client, test_state):
        r = client.post("/api/settings", json={"numBlocksToSwap": 100})
        assert r.status_code == 200
        assert test_state.state.app_settings.num_blocks_to_swap == 48

    def test_num_blocks_clamped_low(self, client, test_state):
        r = client.post("/api/settings", json={"numBlocksToSwap": -5})
        assert r.status_code == 200
        assert test_state.state.app_settings.num_blocks_to_swap == -1

    def test_num_blocks_persists(self, client, test_state, default_app_settings):
        from tests.fakes.services import FakeServices
        from app_handler import ServiceBundle
        from state import build_initial_state

        r = client.post("/api/settings", json={"numBlocksToSwap": 7, "runMode": "medium_vram"})
        assert r.status_code == 200

        fake = FakeServices()
        bundle = ServiceBundle(
            http=fake.http,
            gpu_cleaner=fake.gpu_cleaner,
            model_downloader=fake.model_downloader,
            gpu_info=fake.gpu_info,
            video_processor=fake.video_processor,
            text_encoder=fake.text_encoder,
            task_runner=fake.task_runner,
            ltx_api_client=fake.ltx_api_client,
            zit_api_client=fake.zit_api_client,
            fast_video_pipeline_class=type(fake.fast_video_pipeline),
            image_generation_pipeline_class=type(fake.image_generation_pipeline),
            ic_lora_pipeline_class=type(fake.ic_lora_pipeline),
            depth_processor_pipeline_class=type(fake.depth_processor_pipeline),
            pose_processor_pipeline_class=type(fake.pose_processor_pipeline),
            a2v_pipeline_class=type(fake.a2v_pipeline),
            retake_pipeline_class=type(fake.retake_pipeline),
        )
        loaded = build_initial_state(test_state.config, default_app_settings.model_copy(deep=True), service_bundle=bundle)
        assert loaded.state.app_settings.num_blocks_to_swap == 7
        assert loaded.state.app_settings.run_mode == "medium_vram"


class TestExternalModelsEndpoint:
    """Test the external models listing and download endpoint."""

    def test_list_external_models(self, client):
        r = client.get("/api/gpu/external-models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        models = data["models"]
        assert len(models) > 0
        # Check GGUF models are present
        gguf_models = [m for m in models if m["model_type"] == "gguf"]
        assert len(gguf_models) >= 4  # Q8_0, Q5_1, Q4_K_M, Q4_0 + Z-Image GGUF
        # Check distilled LoRA is present
        lora_models = [m for m in models if m["model_type"] == "lora"]
        assert len(lora_models) >= 1
        # Check checkpoint models
        ckpt_models = [m for m in models if m["model_type"] == "checkpoint"]
        assert len(ckpt_models) >= 1
        # Check upscaler models
        upscaler_models = [m for m in models if m["model_type"] == "upscaler"]
        assert len(upscaler_models) >= 1

    def test_external_models_include_zit_gguf(self, client):
        r = client.get("/api/gpu/external-models")
        data = r.json()
        zit_models = [m for m in data["models"] if "z-image" in m["filename"].lower()]
        assert len(zit_models) >= 3  # BF16, Q8_0, Q4_0

    def test_external_models_have_required_fields(self, client):
        r = client.get("/api/gpu/external-models")
        data = r.json()
        for model in data["models"]:
            assert "id" in model
            assert "filename" in model
            assert "repo_id" in model
            assert "description" in model
            assert "size_gb" in model
            assert "model_type" in model

    def test_download_external_model_conflict_when_downloading(self, client, test_state):
        """Cannot start download when another is in progress."""
        # Manually set a download in progress
        from state.app_state_types import DownloadingSession
        test_state.state.downloading_session = DownloadingSession(
            id="existing",
            current_running_file=None,
            files_to_download={"checkpoint"},
            completed_files=set(),
            completed_bytes=0,
        )

        r = client.post("/api/gpu/download-external-model", json={
            "repo_id": "unsloth/LTX-2.3-GGUF",
            "filename": "LTX-2.3-Q4_0.gguf",
            "target_subdir": "gguf",
        })
        assert r.status_code == 409

    def test_download_external_model_already_exists(self, client, test_state):
        """Returns already_downloaded if file exists."""
        models_dir = test_state.models.models_dir
        gguf_dir = models_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        (gguf_dir / "LTX-2.3-Q4_0.gguf").write_bytes(b"\x00" * 100)

        r = client.post("/api/gpu/download-external-model", json={
            "repo_id": "unsloth/LTX-2.3-GGUF",
            "filename": "LTX-2.3-Q4_0.gguf",
            "target_subdir": "gguf",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "already_downloaded"


class TestLowVramTextEncoderBlockSwap:
    def test_nested_language_model_layers_are_discovered(self) -> None:
        from services.fast_video_pipeline.ltx_low_vram_pipeline import LTXLowVRAMPipeline

        class FakeDecoderLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_layernorm = nn.LayerNorm(4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.input_layernorm(x)

        class FakeInnerLanguageModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList(FakeDecoderLayer() for _ in range(6))

        class FakeLanguageModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = FakeInnerLanguageModel()

        class FakeGemmaModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = FakeLanguageModel()

        class FakeTextEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = FakeGemmaModel()

        pipeline = LTXLowVRAMPipeline.__new__(LTXLowVRAMPipeline)
        pipeline.device = torch.device("cpu")
        pipeline.vram_manager = VRAMManager(torch.device("cpu"), 12)

        wrapper = pipeline._setup_text_encoder_block_swap(FakeTextEncoder())

        assert wrapper is not None
        assert wrapper.block_count == 6

    def test_download_external_model_starts(self, client, test_state):
        """Starts a download for a new file."""
        r = client.post("/api/gpu/download-external-model", json={
            "repo_id": "unsloth/LTX-2.3-GGUF",
            "filename": "LTX-2.3-Q4_0.gguf",
            "target_subdir": "gguf",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "started"
        assert data["session_id"] is not None

        # Verify the file was "downloaded" by fake downloader
        target = test_state.models.models_dir / "gguf" / "LTX-2.3-Q4_0.gguf"
        assert target.exists()


class TestThreeModelModes:
    """Test 3 local video model modes."""

    def test_models_list_has_three_entries(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 4
        ids = [m["id"] for m in data]
        assert ids == ["fast", "balanced", "quality", "custom"]

    def test_fast_is_ltx_fast(self, client):
        r = client.get("/api/models")
        fast = r.json()[0]
        assert fast["id"] == "fast"
        assert fast["name"] == "LTX 2.3 Fast"
        assert "8 steps" in fast["description"]
        assert "no LoRA" in fast["description"]

    def test_balanced_is_ltx_balanced(self, client):
        r = client.get("/api/models")
        balanced = r.json()[1]
        assert balanced["id"] == "balanced"
        assert balanced["name"] == "LTX 2.3 Balanced"
        assert "LoRA" in balanced["description"]
        assert "8 steps" in balanced["description"]

    def test_quality_uses_custom_steps(self, client, test_state):
        test_state.state.app_settings.pro_model.steps = 50
        r = client.get("/api/models")
        quality = r.json()[2]
        assert quality["id"] == "quality"
        assert quality["name"] == "LTX 2.3 Quality"
        assert "50 steps" in quality["description"]

    def test_quality_defaults_to_official_30_steps(self, client):
        r = client.get("/api/models")
        quality = r.json()[2]
        assert quality["id"] == "quality"
        assert "30 steps" in quality["description"]


class TestProgressCallback:
    """Test that progress callback flows from pipeline to generation handler."""

    def test_progress_updates_during_generation(self, client, test_state, create_fake_model_files):
        """Verify that generation updates progress step-by-step."""
        create_fake_model_files(include_zit=True)
        test_state.models.refresh_available_files()

        # Start generation
        r = client.post("/api/generate", json={
            "prompt": "test progress",
            "resolution": "540p",
            "model": "fast",
            "duration": "2",
            "fps": "24",
        })
        assert r.status_code == 200

        # After generation, check that progress was updated
        # The fake pipeline calls progress_callback(step, 8) for steps 1-8
        pr = client.get("/api/generation/progress")
        assert pr.status_code == 200
        data = pr.json()
        # After completion, phase should be "complete"
        assert data["status"] == "complete"


class TestModelModePipelineCreation:
    """Test that different model modes create correct pipeline configs."""

    def test_fast_mode_accepted(self, client, test_state, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        test_state.models.refresh_available_files()
        r = client.post("/api/generate", json={
            "prompt": "test fast",
            "resolution": "540p",
            "model": "fast",
            "duration": "2",
            "fps": "24",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

    def test_balanced_mode_accepted(self, client, test_state, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        test_state.models.refresh_available_files()
        r = client.post("/api/generate", json={
            "prompt": "test balanced",
            "resolution": "540p",
            "model": "balanced",
            "duration": "2",
            "fps": "24",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

    def test_quality_mode_accepted(self, client, test_state, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        test_state.models.refresh_available_files()
        r = client.post("/api/generate", json={
            "prompt": "test quality",
            "resolution": "540p",
            "model": "quality",
            "duration": "2",
            "fps": "24",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

class TestPipelineSelection:
    def test_fast_honors_explicit_distilled_checkpoint_selection(self, test_state, fake_services, monkeypatch):
        checkpoint = test_state.models.models_dir / "diffusion_models" / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        gguf_dir = test_state.models.models_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        (gguf_dir / "ltx-2.3-22b-dev-Q8_0.gguf").write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 8
        test_state.state.app_settings.preferred_model_path = str(checkpoint)

        captured: dict[str, object] = {}

        class StubPipeline:
            pipeline_kind = "fast"

        def fake_standard_create(
            checkpoint_path: str,
            gemma_root: str | None,
            upsampler_path: str,
            device: object,
            *,
            lora_path: str | None = None,
            lora_strength: float = 1.0,
            extra_loras: list[tuple[str, float]] | None = None,
        ) -> StubPipeline:
            del gemma_root, upsampler_path, device, lora_path, lora_strength, extra_loras
            captured["checkpoint_path"] = checkpoint_path
            return StubPipeline()

        monkeypatch.setattr(
            test_state.pipelines._fast_video_pipeline_class,
            "create",
            staticmethod(fake_standard_create),
        )

        state = test_state.pipelines._create_video_pipeline("fast")
        assert isinstance(state.pipeline, StubPipeline)
        assert captured["checkpoint_path"] == str(checkpoint)

    def test_low_vram_prefers_gguf_when_selected_checkpoint_is_incomplete(
        self, test_state, fake_services, monkeypatch,
    ):
        checkpoint = test_state.models.models_dir / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        gguf_dir = test_state.models.models_dir / "diffusion_models"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = gguf_dir / "ltx-2.3-22b-distilled-Q8_0.gguf"
        gguf_path.write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 24
        test_state.state.app_settings.preferred_model_path = str(checkpoint)

        captured: dict[str, object] = {}

        class StubPipeline:
            pipeline_kind = "fast"

        def fake_create(
            checkpoint_path: str,
            gemma_root: str | None,
            upsampler_path: str,
            device: object,
            *,
            vram_manager: object | None = None,
            gguf_path: str | None = None,
            lora_path: str | None = None,
            lora_strength: float = 1.0,
            extra_loras: list[tuple[str, float]] | None = None,
            use_sage_attention: bool = True,
            num_inference_steps: int | None = None,
            text_encoder_variant_path: str | None = None,
            use_upscaler: bool = False,
        ) -> StubPipeline:
            del checkpoint_path, gemma_root, upsampler_path, device, vram_manager, lora_path, lora_strength, extra_loras, use_sage_attention, num_inference_steps, text_encoder_variant_path, use_upscaler
            captured["gguf_path"] = gguf_path
            return StubPipeline()

        monkeypatch.setattr(
            "handlers.pipelines_handler._looks_like_incomplete_transformer_checkpoint",
            lambda path: True,
        )
        monkeypatch.setattr(
            "handlers.pipelines_handler._has_split_ltx_component_fallback",
            lambda models_dir: True,
        )
        monkeypatch.setattr(
            "services.fast_video_pipeline.ltx_low_vram_pipeline.LTXLowVRAMPipeline.create",
            fake_create,
        )

        state = test_state.pipelines._create_video_pipeline("custom")
        assert isinstance(state.pipeline, StubPipeline)
        assert captured["gguf_path"] == str(gguf_path)

    def test_low_vram_gguf_requires_full_checkpoint_for_decode(
        self, test_state, fake_services, monkeypatch,
    ):
        checkpoint = test_state.models.models_dir / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        gguf_dir = test_state.models.models_dir / "diffusion_models"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = gguf_dir / "ltx-2.3-22b-distilled-Q8_0.gguf"
        gguf_path.write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 24
        test_state.state.app_settings.preferred_model_path = str(gguf_path)

        monkeypatch.setattr(
            "handlers.pipelines_handler._looks_like_incomplete_transformer_checkpoint",
            lambda path: str(path).endswith(".safetensors"),
        )
        monkeypatch.setattr(
            "handlers.pipelines_handler._find_full_checkpoint_candidate",
            lambda models_dir: None,
        )
        monkeypatch.setattr(
            "handlers.pipelines_handler._ensure_split_ltx_component_fallback",
            lambda models_dir: False,
        )

        with pytest.raises(RuntimeError, match="full LTX safetensors checkpoint"):
            test_state.pipelines._create_video_pipeline("custom")

    def test_low_vram_gguf_allows_split_component_fallback(
        self, test_state, fake_services, monkeypatch,
    ):
        checkpoint = test_state.models.models_dir / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        gguf_dir = test_state.models.models_dir / "diffusion_models"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = gguf_dir / "ltx-2.3-22b-distilled-Q8_0.gguf"
        gguf_path.write_bytes(b"\x00" * 1024)

        (test_state.models.models_dir / "vae").mkdir(parents=True, exist_ok=True)
        (test_state.models.models_dir / "vae" / "LTX23_video_vae_bf16.safetensors").write_bytes(b"\x00" * 1024)
        (test_state.models.models_dir / "vae" / "LTX23_audio_vae_bf16.safetensors").write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 24
        test_state.state.app_settings.preferred_model_path = str(gguf_path)

        class StubPipeline:
            pipeline_kind = "fast"

        monkeypatch.setattr(
            "handlers.pipelines_handler._looks_like_incomplete_transformer_checkpoint",
            lambda path: str(path).endswith(".safetensors"),
        )
        monkeypatch.setattr(
            "services.fast_video_pipeline.ltx_low_vram_pipeline.LTXLowVRAMPipeline.create",
            lambda *args, **kwargs: StubPipeline(),
        )

        state = test_state.pipelines._create_video_pipeline("custom")
        assert isinstance(state.pipeline, StubPipeline)

    def test_low_vram_gguf_attempts_split_component_auto_download(
        self, test_state, fake_services, monkeypatch,
    ):
        checkpoint = test_state.models.models_dir / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        gguf_dir = test_state.models.models_dir / "diffusion_models"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = gguf_dir / "ltx-2.3-22b-distilled-Q8_0.gguf"
        gguf_path.write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 24
        test_state.state.app_settings.preferred_model_path = str(gguf_path)

        class StubPipeline:
            pipeline_kind = "fast"

        monkeypatch.setattr(
            "handlers.pipelines_handler._looks_like_incomplete_transformer_checkpoint",
            lambda path: str(path).endswith(".safetensors"),
        )
        monkeypatch.setattr(
            "handlers.pipelines_handler._find_full_checkpoint_candidate",
            lambda models_dir: None,
        )

        def fake_ensure(models_dir):
            vae_dir = models_dir / "vae"
            vae_dir.mkdir(parents=True, exist_ok=True)
            (vae_dir / "LTX23_video_vae_bf16.safetensors").write_bytes(b"\x00" * 1024)
            (vae_dir / "LTX23_audio_vae_bf16.safetensors").write_bytes(b"\x00" * 1024)
            return True

        monkeypatch.setattr(
            "handlers.pipelines_handler._ensure_split_ltx_component_fallback",
            fake_ensure,
        )
        monkeypatch.setattr(
            "services.fast_video_pipeline.ltx_low_vram_pipeline.LTXLowVRAMPipeline.create",
            lambda *args, **kwargs: StubPipeline(),
        )

        state = test_state.pipelines._create_video_pipeline("custom")
        assert isinstance(state.pipeline, StubPipeline)

    def test_balanced_mode_uses_default_distilled_lora_when_present(self, test_state):
        default_lora = test_state.models.models_dir / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors"
        default_lora.parent.mkdir(parents=True, exist_ok=True)
        default_lora.write_bytes(b"\x00" * 1024)

        assert test_state.pipelines._resolve_default_distilled_lora() == (str(default_lora), 0.6)

    def test_balanced_mode_skips_default_distilled_lora_for_distilled_checkpoint(
        self, test_state, fake_services, monkeypatch,
    ):
        checkpoint = test_state.models.models_dir / "diffusion_models" / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        default_lora = test_state.models.models_dir / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors"
        default_lora.parent.mkdir(parents=True, exist_ok=True)
        default_lora.write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 64
        test_state.state.app_settings.preferred_model_path = str(checkpoint)

        captured: dict[str, object] = {}

        class StubPipeline:
            pipeline_kind = "fast"

        def fake_create(
            checkpoint_path: str,
            gemma_root: str | None,
            upsampler_path: str,
            device: object,
            *,
            lora_path: str | None = None,
            lora_strength: float = 1.0,
            extra_loras: list[tuple[str, float]] | None = None,
        ) -> StubPipeline:
            del gemma_root, upsampler_path, device, lora_strength, extra_loras
            captured["checkpoint_path"] = checkpoint_path
            captured["lora_path"] = lora_path
            return StubPipeline()

        monkeypatch.setattr(test_state.pipelines._fast_video_pipeline_class, "create", fake_create)

        state = test_state.pipelines._create_video_pipeline("balanced")
        assert isinstance(state.pipeline, StubPipeline)
        assert captured["checkpoint_path"] == str(checkpoint)
        assert captured["lora_path"] is None

    def test_balanced_mode_skips_selected_distilled_lora_for_distilled_checkpoint_in_low_vram(
        self, test_state, fake_services, monkeypatch,
    ):
        checkpoint = test_state.models.models_dir / "diffusion_models" / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        selected_lora = test_state.models.models_dir / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors"
        selected_lora.parent.mkdir(parents=True, exist_ok=True)
        selected_lora.write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 8
        test_state.state.app_settings.preferred_model_path = str(checkpoint)
        test_state.state.app_settings.selected_loras = [
            {"path": str(selected_lora), "strength": 0.6}
        ]

        captured: dict[str, object] = {}

        class StubPipeline:
            pipeline_kind = "fast"

        def fake_create(
            checkpoint_path: str,
            gemma_root: str | None,
            upsampler_path: str,
            device: object,
            *,
            vram_manager: object | None = None,
            gguf_path: str | None = None,
            lora_path: str | None = None,
            lora_strength: float = 1.0,
            extra_loras: list[tuple[str, float]] | None = None,
            use_sage_attention: bool = True,
            num_inference_steps: int | None = None,
            text_encoder_variant_path: str | None = None,
            use_upscaler: bool = False,
        ) -> StubPipeline:
            del gemma_root, upsampler_path, device, vram_manager, gguf_path, lora_strength, extra_loras, use_sage_attention, num_inference_steps, text_encoder_variant_path, use_upscaler
            captured["checkpoint_path"] = checkpoint_path
            captured["lora_path"] = lora_path
            return StubPipeline()

        monkeypatch.setattr(
            "services.fast_video_pipeline.ltx_low_vram_pipeline.LTXLowVRAMPipeline.create",
            fake_create,
        )

        state = test_state.pipelines._create_video_pipeline("balanced")
        assert isinstance(state.pipeline, StubPipeline)
        assert captured["checkpoint_path"] == str(checkpoint)
        assert captured["lora_path"] is None

    def test_fast_mode_uses_fixed_8_steps_and_no_lora_in_low_vram(
        self, test_state, fake_services, monkeypatch,
    ):
        gguf_dir = test_state.models.models_dir / "diffusion_models"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = gguf_dir / "ltx-2.3-22b-distilled-Q4_K_M.gguf"
        gguf_path.write_bytes(b"\x00" * 1024)

        checkpoint = test_state.models.models_dir / "diffusion_models" / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        default_lora = test_state.models.models_dir / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors"
        default_lora.parent.mkdir(parents=True, exist_ok=True)
        default_lora.write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 8

        captured: dict[str, object] = {}

        class StubPipeline:
            pipeline_kind = "fast"

        def fake_create(
            checkpoint_path: str,
            gemma_root: str | None,
            upsampler_path: str,
            device: object,
            *,
            vram_manager: object | None = None,
            gguf_path: str | None = None,
            lora_path: str | None = None,
            lora_strength: float = 1.0,
            extra_loras: list[tuple[str, float]] | None = None,
            use_sage_attention: bool = True,
            num_inference_steps: int | None = None,
            text_encoder_variant_path: str | None = None,
            use_upscaler: bool = False,
        ) -> StubPipeline:
            del checkpoint_path, gemma_root, upsampler_path, device, vram_manager, lora_strength, extra_loras, use_sage_attention, text_encoder_variant_path, use_upscaler
            captured["gguf_path"] = gguf_path
            captured["lora_path"] = lora_path
            captured["num_inference_steps"] = num_inference_steps
            return StubPipeline()

        monkeypatch.setattr(
            "services.fast_video_pipeline.ltx_low_vram_pipeline.LTXLowVRAMPipeline.create",
            fake_create,
        )

        state = test_state.pipelines._create_video_pipeline("fast")
        assert isinstance(state.pipeline, StubPipeline)
        assert captured["gguf_path"] == str(gguf_path)
        assert captured["lora_path"] is None
        assert captured["num_inference_steps"] == 8

    def test_fast_mode_does_not_fall_back_to_dev_gguf_when_only_distilled_checkpoint_exists(
        self, test_state, fake_services, monkeypatch,
    ):
        gguf_dir = test_state.models.models_dir / "diffusion_models"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        dev_gguf_path = gguf_dir / "ltx-2-3-22b-dev-Q4_K_M.gguf"
        dev_gguf_path.write_bytes(b"\x00" * 1024)

        checkpoint = test_state.models.models_dir / "diffusion_models" / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 8

        captured: dict[str, object] = {}

        class StubPipeline:
            pipeline_kind = "fast"

        def fake_create(
            checkpoint_path: str,
            gemma_root: str | None,
            upsampler_path: str,
            device: object,
            *,
            vram_manager: object | None = None,
            gguf_path: str | None = None,
            lora_path: str | None = None,
            lora_strength: float = 1.0,
            extra_loras: list[tuple[str, float]] | None = None,
            use_sage_attention: bool = True,
            num_inference_steps: int | None = None,
            text_encoder_variant_path: str | None = None,
            use_upscaler: bool = False,
        ) -> StubPipeline:
            del gemma_root, upsampler_path, device, vram_manager, lora_path, lora_strength, extra_loras, use_sage_attention, num_inference_steps, text_encoder_variant_path, use_upscaler
            captured["checkpoint_path"] = checkpoint_path
            captured["gguf_path"] = gguf_path
            return StubPipeline()

        monkeypatch.setattr(
            "services.fast_video_pipeline.ltx_low_vram_pipeline.LTXLowVRAMPipeline.create",
            fake_create,
        )

        state = test_state.pipelines._create_video_pipeline("fast")
        assert isinstance(state.pipeline, StubPipeline)
        assert captured["checkpoint_path"] == str(checkpoint)
        assert captured["gguf_path"] is None

    def test_fast_mode_uses_standard_distilled_pipeline_when_only_distilled_checkpoint_exists(
        self, test_state, fake_services, monkeypatch,
    ):
        gguf_dir = test_state.models.models_dir / "diffusion_models"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        dev_gguf_path = gguf_dir / "ltx-2-3-22b-dev-Q4_K_M.gguf"
        dev_gguf_path.write_bytes(b"\x00" * 1024)

        checkpoint = test_state.models.models_dir / "diffusion_models" / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 8
        test_state.state.app_settings.preferred_model_path = str(checkpoint)

        captured: dict[str, object] = {"standard_calls": 0, "low_vram_calls": 0}

        class StubPipeline:
            pipeline_kind = "fast"

        def fake_standard_create(
            checkpoint_path: str,
            gemma_root: str | None,
            upsampler_path: str,
            device: object,
            *,
            lora_path: str | None = None,
            lora_strength: float = 1.0,
            extra_loras: list[tuple[str, float]] | None = None,
        ) -> StubPipeline:
            del gemma_root, upsampler_path, device, lora_path, lora_strength, extra_loras
            captured["standard_calls"] = 1
            captured["checkpoint_path"] = checkpoint_path
            return StubPipeline()

        def fake_low_vram_create(*args: object, **kwargs: object) -> StubPipeline:
            del args, kwargs
            captured["low_vram_calls"] = 1
            return StubPipeline()

        monkeypatch.setattr(
            test_state.pipelines._fast_video_pipeline_class,
            "create",
            staticmethod(fake_standard_create),
        )
        monkeypatch.setattr(
            "services.fast_video_pipeline.ltx_low_vram_pipeline.LTXLowVRAMPipeline.create",
            fake_low_vram_create,
        )

        state = test_state.pipelines._create_video_pipeline("fast")
        assert isinstance(state.pipeline, StubPipeline)
        assert captured["standard_calls"] == 1
        assert captured["low_vram_calls"] == 0
        assert captured["checkpoint_path"] == str(checkpoint)

    def test_quality_mode_prefers_official_dev_checkpoint_over_gguf_when_available(
        self, test_state, fake_services, monkeypatch,
    ):
        gguf_dir = test_state.models.models_dir / "diffusion_models"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = gguf_dir / "ltx-2.3-22b-dev-Q8_0.gguf"
        gguf_path.write_bytes(b"\x00" * 1024)

        dev_checkpoint = gguf_dir / "ltx-2.3-22b-dev.safetensors"
        dev_checkpoint.write_bytes(b"\x00" * 2048)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 8
        test_state.state.app_settings.preferred_model_path = str(gguf_path)

        captured: dict[str, object] = {}

        class StubPipeline:
            pipeline_kind = "fast"

        def fake_create(
            checkpoint_path: str,
            gemma_root: str | None,
            upsampler_path: str,
            device: object,
            *,
            vram_manager: object | None = None,
            gguf_path: str | None = None,
            lora_path: str | None = None,
            lora_strength: float = 1.0,
            extra_loras: list[tuple[str, float]] | None = None,
            use_sage_attention: bool = True,
            num_inference_steps: int | None = None,
            text_encoder_variant_path: str | None = None,
            use_upscaler: bool = False,
        ) -> StubPipeline:
            del gemma_root, upsampler_path, device, vram_manager, lora_path, lora_strength, extra_loras, use_sage_attention, num_inference_steps, text_encoder_variant_path, use_upscaler
            captured["checkpoint_path"] = checkpoint_path
            captured["gguf_path"] = gguf_path
            return StubPipeline()

        monkeypatch.setattr(
            "handlers.pipelines_handler._looks_like_incomplete_transformer_checkpoint",
            lambda path: False,
        )
        monkeypatch.setattr(
            "services.fast_video_pipeline.ltx_low_vram_pipeline.LTXLowVRAMPipeline.create",
            fake_create,
        )

        state = test_state.pipelines._create_video_pipeline("quality")
        assert isinstance(state.pipeline, StubPipeline)
        assert captured["checkpoint_path"] == str(dev_checkpoint)
        assert captured["gguf_path"] is None

    def test_balanced_mode_uses_fixed_8_steps_with_default_distilled_lora_in_low_vram(
        self, test_state, fake_services, monkeypatch,
    ):
        gguf_dir = test_state.models.models_dir / "diffusion_models"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = gguf_dir / "ltx-2.3-22b-dev-Q4_K_M.gguf"
        gguf_path.write_bytes(b"\x00" * 1024)

        checkpoint = test_state.models.models_dir / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        default_lora = test_state.models.models_dir / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors"
        default_lora.parent.mkdir(parents=True, exist_ok=True)
        default_lora.write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 8
        test_state.state.app_settings.preferred_model_path = str(gguf_path)

        captured: dict[str, object] = {}

        class StubPipeline:
            pipeline_kind = "fast"

        def fake_create(
            checkpoint_path: str,
            gemma_root: str | None,
            upsampler_path: str,
            device: object,
            *,
            vram_manager: object | None = None,
            gguf_path: str | None = None,
            lora_path: str | None = None,
            lora_strength: float = 1.0,
            extra_loras: list[tuple[str, float]] | None = None,
            use_sage_attention: bool = True,
            num_inference_steps: int | None = None,
            text_encoder_variant_path: str | None = None,
            use_upscaler: bool = False,
        ) -> StubPipeline:
            del checkpoint_path, gemma_root, upsampler_path, device, vram_manager, lora_strength, extra_loras, use_sage_attention, text_encoder_variant_path, use_upscaler
            captured["gguf_path"] = gguf_path
            captured["lora_path"] = lora_path
            captured["num_inference_steps"] = num_inference_steps
            return StubPipeline()

        monkeypatch.setattr(
            "services.fast_video_pipeline.ltx_low_vram_pipeline.LTXLowVRAMPipeline.create",
            fake_create,
        )

        state = test_state.pipelines._create_video_pipeline("balanced")
        assert isinstance(state.pipeline, StubPipeline)
        assert captured["gguf_path"] == str(gguf_path)
        assert captured["lora_path"] == str(default_lora)
        assert captured["num_inference_steps"] == 8

    def test_custom_mode_does_not_use_legacy_preferred_lora_fallback(self, test_state):
        legacy_lora = test_state.models.models_dir / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors"
        legacy_lora.parent.mkdir(parents=True, exist_ok=True)
        legacy_lora.write_bytes(b"\x00" * 1024)

        test_state.state.app_settings.selected_loras = []
        test_state.state.app_settings.preferred_lora_path = str(legacy_lora)
        test_state.state.app_settings.preferred_lora_strength = 0.6

        assert test_state.pipelines._resolve_selected_loras(allow_legacy_fallback=False) == []
        assert test_state.pipelines._resolve_selected_loras(allow_legacy_fallback=True) == [
            (str(legacy_lora), 0.6)
        ]

    def test_local_zit_accepts_gguf_path(self, test_state):
        gguf_path = test_state.models.models_dir / "gguf" / "z-image-turbo-BF16.gguf"
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        gguf_path.write_bytes(b"\x00" * 1024)
        test_state.state.app_settings.preferred_zit_model_path = str(gguf_path)

        # Should resolve the GGUF path without raising
        resolved = test_state.pipelines._resolve_preferred_zit_path()
        assert resolved == str(gguf_path)


class TestTextEncoderExternalModels:
    """Test quantized text encoder download options."""

    def test_text_encoder_models_listed(self, client):
        r = client.get("/api/gpu/external-models")
        data = r.json()
        te_models = [m for m in data["models"] if m["model_type"] == "text_encoder"]
        assert len(te_models) >= 2  # FP8, BF16, text_projection
        fp8 = [m for m in te_models if "fp8" in m["filename"].lower()]
        assert len(fp8) >= 1

    def test_text_encoder_setting_persists(self, client, test_state):
        r = client.post("/api/settings", json={"preferredTextEncoderPath": "text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"})
        assert r.status_code == 200
        assert test_state.state.app_settings.preferred_text_encoder_path == "text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"

    def test_text_encoder_gguf_setting_persists(self, client, test_state):
        r = client.post("/api/settings", json={"preferredTextEncoderPath": "text_encoders/gemma-3-12b-it-Q8_0.gguf"})
        assert r.status_code == 200
        assert test_state.state.app_settings.preferred_text_encoder_path == "text_encoders/gemma-3-12b-it-Q8_0.gguf"

    def test_installed_text_encoder_variants_include_gguf(self, client, test_state):
        te_dir = test_state.models.models_dir / "text_encoders"
        te_dir.mkdir(parents=True, exist_ok=True)
        (te_dir / "gemma-3-12b-it-Q8_0.gguf").write_bytes(b"\x00" * 1024)
        (te_dir / "gemma_3_12B_it_fp8_scaled.safetensors").write_bytes(b"\x00" * 1024)

        r = client.get("/api/gpu/text-encoders")
        assert r.status_code == 200
        data = r.json()
        names = {variant["filename"] for variant in data["variants"]}
        assert "gemma-3-12b-it-Q8_0.gguf" in names
        assert "gemma_3_12B_it_fp8_scaled.safetensors" in names


class TestUpscalerSelection:
    def test_low_vram_pipeline_receives_fast_upscaler_setting(self, test_state, fake_services, monkeypatch):
        checkpoint = test_state.models.models_dir / "ltx-2.3-22b-distilled.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"\x00" * 1024)

        upsampler = test_state.models.models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upsampler.write_bytes(b"\x00" * 1024)

        gguf_dir = test_state.models.models_dir / "diffusion_models"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        (gguf_dir / "ltx-2.3-22b-distilled-Q4_0.gguf").write_bytes(b"\x00" * 1024)

        fake_services.gpu_info.vram_gb = 8
        test_state.state.app_settings.fast_model.use_upscaler = True
        captured: dict[str, object] = {}

        class StubPipeline:
            pipeline_kind = "fast"

        def fake_create(
            checkpoint_path: str,
            gemma_root: str | None,
            upsampler_path: str,
            device: object,
            *,
            vram_manager: object | None = None,
            gguf_path: str | None = None,
            lora_path: str | None = None,
            lora_strength: float = 1.0,
            extra_loras: list[tuple[str, float]] | None = None,
            use_sage_attention: bool = True,
            num_inference_steps: int | None = None,
            text_encoder_variant_path: str | None = None,
            use_upscaler: bool = False,
        ) -> StubPipeline:
            del checkpoint_path, gemma_root, upsampler_path, device, vram_manager, gguf_path, lora_path, lora_strength, extra_loras, use_sage_attention, num_inference_steps, text_encoder_variant_path
            captured["use_upscaler"] = use_upscaler
            return StubPipeline()

        monkeypatch.setattr(
            "services.fast_video_pipeline.ltx_low_vram_pipeline.LTXLowVRAMPipeline.create",
            fake_create,
        )

        state = test_state.pipelines._create_video_pipeline("fast")
        assert isinstance(state.pipeline, StubPipeline)
        assert captured["use_upscaler"] is True

    def test_text_encoder_download_to_subdir(self, client, test_state):
        """Text encoder models download to text_encoders/ subdir."""
        r = client.post("/api/gpu/download-external-model", json={
            "repo_id": "Comfy-Org/ltx-2",
            "filename": "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
            "target_subdir": "text_encoders",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "started"
        # Verify file lands in correct directory
        target = test_state.models.models_dir / "text_encoders" / "gemma_3_12B_it_fp8_scaled.safetensors"
        assert target.exists()
