"""Retake API orchestration handler."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
import time

from api_types import GenerationProgressResponse, RetakeRequest, RetakeResponse
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from runtime_config.runtime_config import RuntimeConfig
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from services.interfaces import LTXAPIClient
from state.app_state_types import AppState
from state.app_settings import should_video_generate_with_ltx_api
from server_utils.ltx_video_normalization import downsample_video_temporally_for_ltx, normalize_video_for_ltx
from services.vram_manager.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


class RetakeHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        ltx_api_client: LTXAPIClient,
        config: RuntimeConfig,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
    ) -> None:
        super().__init__(state, lock, config)
        self._ltx_api_client = ltx_api_client
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler

    def run(self, req: RetakeRequest) -> RetakeResponse:
        video_path = req.video_path
        start_time = req.start_time
        duration = req.duration
        prompt = req.prompt
        mode = req.mode
        resolution = req.resolution

        if not video_path:
            raise HTTPError(400, "Missing video_path parameter")
        if duration < 2:
            raise HTTPError(400, "duration must be at least 2 seconds")

        video_file = Path(video_path)
        if not video_file.exists():
            raise HTTPError(400, f"Video file not found: {video_path}")

        use_api = should_video_generate_with_ltx_api(
            force_api_generations=self.config.force_api_generations,
            settings=self.state.app_settings,
        )

        if use_api:
            return self._run_api_retake(
                video_file=video_file,
                start_time=start_time,
                duration=duration,
                prompt=prompt,
                mode=mode,
            )

        normalized_video_file = self._normalize_input_video(video_file, keep_audio=True, resolution=resolution)

        return self._run_local_retake(
            video_file=normalized_video_file,
            start_time=start_time,
            duration=duration,
            prompt=prompt,
            mode=mode,
        )

    def _run_api_retake(
        self,
        *,
        video_file: Path,
        start_time: float,
        duration: float,
        prompt: str,
        mode: str,
    ) -> RetakeResponse:
        api_key = self.state.app_settings.ltx_api_key
        if not api_key:
            raise HTTPError(400, "LTX API key not configured. Set it in Settings.")

        try:
            result = self._ltx_api_client.retake(
                api_key=api_key,
                video_path=str(video_file),
                start_time=start_time,
                duration=duration,
                prompt=prompt,
                mode=mode,
            )
        except LTXAPIClientError as exc:
            raise HTTPError(exc.status_code, exc.detail) from exc

        if result.video_bytes is not None:
            output = self.config.outputs_dir / f"retake_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
            with open(output, "wb") as out:
                out.write(result.video_bytes)
            return RetakeResponse(status="complete", video_path=str(output))

        if result.result_payload is not None:
            return RetakeResponse(status="complete", result=result.result_payload)

        raise HTTPError(500, "Retake API returned no result")

    def _run_local_retake(
        self,
        *,
        video_file: Path,
        start_time: float,
        duration: float,
        prompt: str,
        mode: str,
    ) -> RetakeResponse:
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        end_time = start_time + duration
        if start_time >= end_time:
            raise HTTPError(400, "start_time must be less than end_time")

        self._validate_video_metadata(str(video_file))

        try:
            from ltx_pipelines.utils.media_io import get_videostream_metadata

            fps, num_frames, _, _ = get_videostream_metadata(str(video_file))
            max_duration = max(0.0, float(num_frames) / float(fps or 1.0))
            if start_time >= max_duration:
                start_time = max(0.0, max_duration - duration)
            end_time = min(end_time, max_duration)
        except Exception:
            logger.warning("Failed to clamp retake times to normalized video duration", exc_info=True)

        try:
            self._text.prepare_text_encoding(prompt, enhance_prompt=False)
        except RuntimeError as exc:
            raise HTTPError(400, str(exc)) from exc

        generation_id = uuid.uuid4().hex[:8]
        seed = self._resolve_seed()
        output_path = self.config.outputs_dir / f"retake_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{generation_id}.mp4"
        regenerate_video, regenerate_audio = self._resolve_retake_mode(mode)

        try:
            pipeline_state = self._pipelines.load_retake_pipeline(distilled=True)
            self._generation.start_generation(generation_id)
            self._generation.update_progress("loading_model", 5, 0, 1)
            self._generation.update_progress("inference", 15, 0, 1)

            def on_step(current_step: int, total_steps: int) -> None:
                step_fraction = current_step / max(total_steps, 1)
                progress = 15 + int(step_fraction * 75)
                logger.info("[retake] Denoise step %d/%d", current_step, total_steps)
                self._generation.update_progress("inference", progress, current_step, total_steps)
                self._generation._live_progress_snapshot = GenerationProgressResponse(  # type: ignore[attr-defined]
                    status="running",
                    phase="inference",
                    progress=progress,
                    currentStep=current_step,
                    totalSteps=total_steps,
                )
                import handlers.generation_handler as generation_handler_module
                generation_handler_module._GLOBAL_LIVE_PROGRESS_SNAPSHOT = GenerationProgressResponse(
                    status="running",
                    phase="inference",
                    progress=progress,
                    currentStep=current_step,
                    totalSteps=total_steps,
                )

            pipeline_state.pipeline.generate(
                video_path=str(video_file),
                prompt=prompt,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
                output_path=str(output_path),
                negative_prompt=self.config.default_negative_prompt,
                num_inference_steps=40,
                video_guider_params=None,
                audio_guider_params=None,
                regenerate_video=regenerate_video,
                regenerate_audio=regenerate_audio,
                enhance_prompt=False,
                distilled=True,
                progress_callback=on_step,
            )

            if self._generation.is_generation_cancelled():
                output_path.unlink(missing_ok=True)
                raise RuntimeError("Generation was cancelled")

            self._generation.update_progress("complete", 100, 1, 1)
            self._generation.complete_generation(str(output_path))
            return RetakeResponse(status="complete", video_path=str(output_path))
        except HTTPError:
            self._generation.fail_generation("Retake generation failed")
            raise
        except Exception as exc:
            self._generation.fail_generation(str(exc))
            if "cancelled" in str(exc).lower():
                return RetakeResponse(status="cancelled")
            raise HTTPError(500, f"Generation error: {exc}") from exc
        finally:
            self._text.clear_api_embeddings()

    @staticmethod
    def _resolve_retake_mode(mode: str) -> tuple[bool, bool]:
        if mode == "replace_audio_and_video":
            return True, True
        if mode in {"replace_video", "replace_video_only"}:
            return True, False
        if mode == "replace_audio":
            return False, True
        raise HTTPError(400, "INVALID_RETAKE_MODE")

    def _resolve_seed(self) -> int:
        settings = self.state.app_settings
        if settings.seed_locked:
            return settings.locked_seed
        return int(time.time()) % 2147483647

    def _normalize_input_video(self, video_file: Path, *, keep_audio: bool, resolution: str) -> Path:
        from ltx_pipelines.utils.media_io import get_videostream_metadata

        fps, num_frames, width, height = get_videostream_metadata(str(video_file))
        target_width, target_height = self._resolve_target_resolution(
            resolution,
            source_width=int(width),
            source_height=int(height),
        )
        normalized = normalize_video_for_ltx(
            video_path=str(video_file),
            output_dir=self.config.outputs_dir / "_normalized_inputs",
            fps=float(fps),
            num_frames=int(num_frames),
            width=int(width),
            height=int(height),
            keep_audio=keep_audio,
            target_width=target_width,
            target_height=target_height,
        )
        current_path = Path(normalized.path) if normalized.changed else video_file
        if normalized.changed:
            logger.info(
                "Normalized retake input %s -> %s (frames %d->%d size %dx%d->%dx%d)",
                video_file,
                normalized.path,
                normalized.original_frames,
                normalized.normalized_frames,
                normalized.original_width,
                normalized.original_height,
                normalized.normalized_width,
                normalized.normalized_height,
            )

        import torch

        vram_gb = 0
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)  # pyright: ignore[reportUnknownMemberType]
            vram_gb = int((int(props.total_memory) + (1024**3 - 1)) // (1024**3))  # pyright: ignore[reportUnknownMemberType]

        manager = VRAMManager(
            self.config.device,
            vram_gb,
            user_blocks_on_gpu=self.state.app_settings.num_blocks_to_swap,
            user_run_mode=self.state.app_settings.run_mode,
        )
        max_frames = manager.get_max_frames(
            normalized.normalized_width,
            normalized.normalized_height,
            max(1, int(round(normalized.fps))),
        )
        if normalized.normalized_frames > max_frames:
            temporal = downsample_video_temporally_for_ltx(
                video_path=str(current_path),
                output_dir=self.config.outputs_dir / "_normalized_inputs",
                target_max_frames=max_frames,
            )
            logger.info(
                "Temporally downsampled retake input %s -> %s (frames %d->%d fps %.2f->%.2f)",
                current_path,
                temporal.path,
                temporal.original_frames,
                temporal.normalized_frames,
                normalized.fps,
                temporal.fps,
            )
            return Path(temporal.path)

        return current_path

    @staticmethod
    def _resolve_target_resolution(resolution: str, *, source_width: int, source_height: int) -> tuple[int, int]:
        landscape_sizes: dict[str, tuple[int, int]] = {
            "540p": (960, 544),
            "720p": (1280, 704),
            "1080p": (1920, 1088),
        }
        base_width, base_height = landscape_sizes.get(resolution, landscape_sizes["540p"])
        if source_height > source_width:
            return base_height, base_width
        return base_width, base_height

    @staticmethod
    def _validate_video_metadata(video_path: str) -> None:
        from ltx_core.types import SpatioTemporalScaleFactors
        from ltx_pipelines.utils.media_io import get_videostream_metadata

        fps, num_frames, width, height = get_videostream_metadata(video_path)
        del fps
        scale = SpatioTemporalScaleFactors.default()
        if (num_frames - 1) % scale.time != 0:
            snapped = ((num_frames - 1) // scale.time) * scale.time + 1
            raise HTTPError(
                400,
                f"Video frame count must satisfy 8k+1 (e.g. 97, 193). Got {num_frames}; use a video with {snapped} frames.",
            )
        if width % 32 != 0 or height % 32 != 0:
            raise HTTPError(400, f"Video width and height must be multiples of 32. Got {width}x{height}.")
