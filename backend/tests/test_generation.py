"""Integration-style tests for generation and image endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from state.app_settings import SelectedLoRASettings
from state.app_state_types import GpuSlot, VideoPipelineState, VideoPipelineWarmth
from services.vram_manager.vram_manager import OffloadStrategy
from tests.fakes.services import FakeFastVideoPipeline


@dataclass
class _FakeEncodingResult:
    """Minimal stand-in for TextEncodingResult in tests."""

    video_context: object = "fake_tensor"
    audio_context: object = None

_T2V_JSON = {
    "prompt": "test",
    "resolution": "540p",
    "model": "fast",
    "duration": "2",
    "fps": "24",
}


def _write_test_wav(path: Path, *, duration_seconds: float = 0.1, sample_rate: int = 8000) -> None:
    import wave

    frame_count = max(1, int(duration_seconds * sample_rate))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def _enable_local_text_encoding(test_state) -> None:
    test_state.state.app_settings.use_local_text_encoder = True


def _write_fake_a2v_model_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 1024)


def _configure_a2v_fast_distilled_inputs(test_state) -> None:
    models_dir = test_state.config.default_models_dir
    distilled_gguf = models_dir / "diffusion_models" / "ltx-2.3-22b-distilled-Q4_0.gguf"
    _write_fake_a2v_model_file(distilled_gguf)
    test_state.state.app_settings.preferred_model_path = str(distilled_gguf)
    test_state.state.app_settings.preferred_gguf_path = ""
    test_state.state.app_settings.selected_loras = []
    test_state.state.app_settings.preferred_lora_path = ""
    test_state.state.app_settings.preferred_lora_strength = 1.0


def _configure_a2v_balanced_dev_distilled_lora_inputs(test_state) -> tuple[Path, Path]:
    models_dir = test_state.config.default_models_dir
    dev_gguf = models_dir / "diffusion_models" / "ltx-2.3-22b-dev-Q8_0.gguf"
    distilled_lora = models_dir / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors"
    _write_fake_a2v_model_file(dev_gguf)
    _write_fake_a2v_model_file(distilled_lora)
    test_state.state.app_settings.preferred_model_path = str(dev_gguf)
    test_state.state.app_settings.preferred_gguf_path = ""
    test_state.state.app_settings.selected_loras = [
        SelectedLoRASettings(path=str(distilled_lora), strength=0.6)
    ]
    test_state.state.app_settings.preferred_lora_path = str(distilled_lora)
    test_state.state.app_settings.preferred_lora_strength = 0.6
    return dev_gguf, distilled_lora


def _fake_running_generation_state(test_state) -> None:
    pipeline = FakeFastVideoPipeline()
    test_state.state.gpu_slot = GpuSlot(
        active_pipeline=VideoPipelineState(
            pipeline=pipeline,
            warmth=VideoPipelineWarmth.COLD,
            is_compiled=False,
        ),
        generation=None,
    )
    test_state.generation.start_generation("running")


class TestGenerate:
    def test_t2v_happy_path(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A beautiful sunset",
                "resolution": "1080p",
                "model": "fast",
                "duration": "2",
                "fps": "24",
                "cameraMotion": "none",
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"] is not None
        assert Path(data["video_path"]).exists()

        pipeline = fake_services.fast_video_pipeline
        assert len(pipeline.generate_calls) == 1

    def test_t2v_clears_vram_after_success(self, client, test_state, fake_services, create_fake_model_files):
        """GPU pipeline is unloaded after a successful T2V generation."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

        # The GPU slot should have been cleared
        assert test_state.state.gpu_slot is None
        # GPU cleaner should have been invoked
        assert fake_services.gpu_cleaner.cleanup_calls >= 1

    def test_a2v_clears_vram_after_success(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        """GPU pipeline is unloaded after a successful A2V generation."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        _configure_a2v_fast_distilled_inputs(test_state)

        audio = tmp_path / "audio.wav"
        _write_test_wav(audio, duration_seconds=2.0)

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "audioPath": str(audio),
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

        # The GPU slot should have been cleared
        assert test_state.state.gpu_slot is None
        assert fake_services.gpu_cleaner.cleanup_calls >= 1

    def test_already_running(self, client, test_state):
        _fake_running_generation_state(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 409

    def test_i2v_nonexistent_image(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "imagePath": "/no/such/file.png"},
        )
        assert r.status_code == 400

    def test_i2v_rejects_invalid_image_content_400(self, client, test_state, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        bad_image = tmp_path / "bad.png"
        bad_image.write_bytes(b"not-a-real-png")

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "imagePath": str(bad_image)},
        )
        assert r.status_code == 400
        assert "Invalid image file" in r.json()["error"]

    def test_resolution_mapping_540p(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["width"] == 960
        assert call["height"] == 512

    def test_resolution_mapping_720p(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json={**_T2V_JSON, "resolution": "720p"})
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["width"] == 1280
        assert call["height"] == 704

    def test_locked_seed(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 123

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        assert pipeline.generate_calls[0]["seed"] == 123

    def test_negative_prompt_is_forwarded_to_pipeline(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "negativePrompt": "low quality, artifacts"},
        )
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        assert pipeline.generate_calls[0]["negative_prompt"] == "low quality, artifacts"

    def test_experimental_three_stage_requires_image(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={**_T2V_JSON, "advancedMode": "experimental_three_stage_sampling"},
        )

        assert r.status_code == 400
        assert "requires an input image" in r.json()["error"]

    def test_experimental_three_stage_rejects_until_executor_is_wired(
        self, client, test_state, create_fake_model_files, tmp_path
    ):
        from PIL import Image

        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        image_path = tmp_path / "input.png"
        Image.new("RGB", (64, 64), "white").save(image_path)

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "advancedMode": "experimental_three_stage_sampling",
                "imagePath": str(image_path),
            },
        )

        assert r.status_code == 400
        assert "standalone workflow asset" in r.json()["error"]

    def test_error_sets_generation_error(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        fake_services.fast_video_pipeline.raise_on_generate = RuntimeError("GPU OOM")

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 500

        progress = test_state.generation.get_generation_progress()
        assert progress.status == "error"

    def test_oom_recovers_gpu_state_and_clears_text_encoder_cache(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        class _CachedEncoder:
            def __init__(self) -> None:
                self.moves: list[str] = []

            def to(self, device):
                self.moves.append(str(device))
                return self

        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        fake_services.fast_video_pipeline.raise_on_generate = RuntimeError("CUDA out of memory")
        cached_encoder = _CachedEncoder()
        test_state.state.text_encoder.cached_encoder = cached_encoder
        test_state.state.text_encoder.prompt_cache[("prompt", False)] = _FakeEncodingResult()

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 500

        assert test_state.state.gpu_slot is None
        assert test_state.state.text_encoder.cached_encoder is None
        assert test_state.state.text_encoder.prompt_cache == {}
        assert cached_encoder.moves == ["cpu"]
        assert fake_services.gpu_cleaner.cleanup_calls >= 1

    def test_cancelled_response(self, client, test_state, fake_services, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        fake_services.fast_video_pipeline.raise_on_generate = RuntimeError("cancelled")

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


class TestA2VGenerate:
    def test_a2v_generation_happy_path(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "resolution": "540p",
                "model": "fast",
                "duration": "2",
                "fps": "24",
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"] is not None
        assert Path(data["video_path"]).exists()

        pipeline = fake_services.a2v_pipeline
        assert len(pipeline.create_calls) == 1
        assert len(pipeline.generate_calls) == 1
        create_call = pipeline.create_calls[0]
        call = pipeline.generate_calls[0]
        assert create_call["vram_manager"] is not None
        assert create_call["use_sage_attention"] is False
        assert create_call["vram_manager"].offload_strategy in (
            OffloadStrategy.BLOCK_SWAP,
            OffloadStrategy.BLOCK_SWAP_AGGRESSIVE,
        )
        assert call["audio_path"] == str(audio_file)
        assert call["audio_start_time"] == 0.0
        assert call["audio_max_duration"] is None
        assert callable(call["progress_callback"])

    def test_a2v_rejects_missing_audio_file(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "duration": "2",
                "fps": "24",
                "audioPath": "/no/such/audio.wav",
            },
        )
        assert r.status_code == 400

    def test_a2v_rejects_invalid_audio_content_400(self, client, test_state, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "bad.wav"
        audio_file.write_bytes(b"not-a-real-wav")

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "duration": "2",
                "fps": "24",
                "audioPath": str(audio_file),
            },
        )
        assert r.status_code == 400
        assert "Invalid audio file" in r.json()["error"]

    def test_a2v_uses_resolution_map(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        for resolution, expected_w, expected_h in [
            ("540p", 960, 576),
            ("720p", 1280, 704),
            ("1080p", 1920, 1088),
        ]:
            fake_services.a2v_pipeline.generate_calls.clear()
            r = client.post(
                "/api/generate",
                json={
                    "prompt": "A music video",
                    "resolution": resolution,
                    "model": "pro",
                    "duration": "2",
                    "fps": "24",
                    "audioPath": str(audio_file),
                },
            )

            assert r.status_code == 200
            call = fake_services.a2v_pipeline.generate_calls[0]
            assert call["width"] == expected_w, f"{resolution}: expected width {expected_w}, got {call['width']}"
            assert call["height"] == expected_h, f"{resolution}: expected height {expected_h}, got {call['height']}"

    def test_a2v_supports_portrait_9_16_resolution_map(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A portrait music video",
                "resolution": "1080p",
                "aspectRatio": "9:16",
                "model": "fast",
                "duration": "2",
                "fps": "24",
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        call = fake_services.a2v_pipeline.generate_calls[0]
        assert call["width"] == 1088
        assert call["height"] == 1920

    def test_a2v_fast_prefers_distilled_model_without_lora(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        _configure_a2v_fast_distilled_inputs(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "model": "fast",
                "duration": "2",
                "fps": "24",
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        create_call = fake_services.a2v_pipeline.create_calls[0]
        assert create_call["gguf_path"] is not None
        assert "distilled" in create_call["gguf_path"].lower()
        assert create_call["lora_path"] is None
        assert create_call["num_inference_steps"] == 8

    def test_a2v_balanced_prefers_dev_plus_distilled_lora(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        dev_gguf, distilled_lora = _configure_a2v_balanced_dev_distilled_lora_inputs(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "model": "balanced",
                "duration": "2",
                "fps": "24",
                "audioPath": str(audio_file),
            },
        )

        assert r.status_code == 200
        create_call = fake_services.a2v_pipeline.create_calls[0]
        assert create_call["gguf_path"] == str(dev_gguf)
        assert create_call["lora_path"] == str(distilled_lora)
        assert create_call["lora_strength"] == 0.6
        assert create_call["num_inference_steps"] == 8

    def test_a2v_fast_reloads_pipeline_after_quality(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files()
        _enable_local_text_encoding(test_state)
        _configure_a2v_fast_distilled_inputs(test_state)
        audio_file = tmp_path / "test_audio.wav"
        _write_test_wav(audio_file)

        quality = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "model": "pro",
                "duration": "2",
                "fps": "24",
                "audioPath": str(audio_file),
            },
        )
        assert quality.status_code == 200

        fast = client.post(
            "/api/generate",
            json={
                "prompt": "A music video",
                "model": "fast",
                "duration": "2",
                "fps": "24",
                "audioPath": str(audio_file),
            },
        )

        assert fast.status_code == 200
        assert len(fake_services.a2v_pipeline.create_calls) == 2
        fast_create_call = fake_services.a2v_pipeline.create_calls[1]
        assert fast_create_call["gguf_path"] is not None
        assert "distilled" in fast_create_call["gguf_path"].lower()
        assert fast_create_call["lora_path"] is None
        assert fast_create_call["num_inference_steps"] == 8

class TestGenerateCancel:
    def test_cancel_active(self, client, test_state):
        _fake_running_generation_state(test_state)

        r = client.post("/api/generate/cancel")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "cancelling"

    def test_cancel_no_active(self, client):
        r = client.post("/api/generate/cancel")
        assert r.status_code == 200
        assert r.json()["status"] == "no_active_generation"


class TestGenerationProgress:
    def test_idle(self, client):
        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        assert r.json()["status"] == "idle"

    def test_running(self, client, test_state):
        _fake_running_generation_state(test_state)
        test_state.generation.update_progress("inference", 50, 4, 8)

        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"
        assert data["phase"] == "inference"
        assert data["progress"] == 50
        assert data["currentStep"] == 4
        assert data["totalSteps"] == 8

    def test_running_from_api_generation_state(self, client, test_state):
        test_state.generation.start_api_generation("api-running")
        test_state.generation.update_progress("inference", 35)

        r = client.get("/api/generation/progress")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"
        assert data["phase"] == "inference"
        assert data["progress"] == 35
        assert data["currentStep"] is None
        assert data["totalSteps"] is None


class TestGenerateImage:
    def test_happy_path(self, client, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "A cat", "width": 1024, "height": 1024, "numSteps": 4},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert len(data["image_paths"]) == 1
        assert Path(data["image_paths"][0]).exists()

    def test_dimension_clamping(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "width": 1023, "height": 1023},
        )
        assert r.status_code == 200

        call = fake_services.image_generation_pipeline.generate_calls[0]
        assert call["width"] == 1008
        assert call["height"] == 1008

    def test_num_images_clamped(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "numImages": 20},
        )
        assert r.status_code == 200

        assert len(fake_services.image_generation_pipeline.generate_calls) == 12

    def test_error(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        fake_services.image_generation_pipeline.raise_on_generate = RuntimeError("GPU OOM")

        r = client.post("/api/generate-image", json={"prompt": "test"})
        assert r.status_code == 500

    def test_cancelled(self, client, fake_services, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        fake_services.image_generation_pipeline.raise_on_generate = RuntimeError("cancelled")

        r = client.post("/api/generate-image", json={"prompt": "test"})
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


class TestEmptyPromptRejected:
    def test_empty_prompt_rejected(self, client):
        r = client.post("/api/generate", json={"prompt": ""})
        assert r.status_code == 422

    def test_whitespace_prompt_rejected(self, client):
        r = client.post("/api/generate", json={"prompt": "   "})
        assert r.status_code == 422

    def test_missing_prompt_rejected(self, client):
        r = client.post("/api/generate", json={})
        assert r.status_code == 422

    def test_empty_image_prompt_rejected(self, client):
        r = client.post("/api/generate-image", json={"prompt": ""})
        assert r.status_code == 422

    def test_whitespace_image_prompt_rejected(self, client):
        r = client.post("/api/generate-image", json={"prompt": "   "})
        assert r.status_code == 422

    def test_missing_image_prompt_rejected(self, client):
        r = client.post("/api/generate-image", json={})
        assert r.status_code == 422
