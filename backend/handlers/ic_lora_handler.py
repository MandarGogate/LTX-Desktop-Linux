"""IC-LoRA endpoints orchestration handler."""

from __future__ import annotations

import base64
import logging
import math
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, cast

import torch

from api_types import (
    IcLoraExtractRequest,
    IcLoraExtractResponse,
    IcLoraGenerateRequest,
    IcLoraGenerateResponse,
    ImageConditioningInput,
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import resolve_model_path
from runtime_config.runtime_config import RuntimeConfig
from state.conditioning_cache import ConditioningCacheEntry, ConditioningCacheKey
from services.interfaces import VideoProcessor
from services.services_utils import FrameArray
from services.vram_manager.vram_manager import VRAMManager
from server_utils.ltx_video_normalization import downsample_video_temporally_for_ltx, snap_frames_to_8k_plus_1
from server_utils.motion_track_overlay import create_motion_track_overlay_video
from state.app_state_types import AppState, ICLoraState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


_UNION_CONDITIONING_TYPES = {"canny", "depth", "pose"}


class IcLoraHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        video_processor: VideoProcessor,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._video_processor = video_processor

    def _build_conditioning_frame(
        self,
        frame: FrameArray,
        conditioning_type: str,
        ic_state: ICLoraState | None = None,
    ) -> FrameArray:
        match conditioning_type:
            case "canny":
                return self._video_processor.apply_canny(frame)
            case "depth":
                if ic_state is None or ic_state.depth_pipeline is None:
                    raise HTTPError(500, "Depth conditioning requires loaded IC-LoRA resources")
                return self._video_processor.apply_depth(frame, ic_state.depth_pipeline)
            case "pose":
                if ic_state is None or ic_state.pose_pipeline is None:
                    raise HTTPError(500, "Pose conditioning requires loaded IC-LoRA resources")
                return self._video_processor.apply_pose(frame, ic_state.pose_pipeline)
            case "motion_track":
                return frame
            case _:
                raise HTTPError(400, f"Unsupported conditioning_type: {conditioning_type}")

    def _resolve_ic_lora_resources(
        self,
        *,
        model_type: str,
        conditioning_type: str,
    ) -> tuple[Path, Path | None, Path | None, Path | None]:
        if model_type == "motion_track":
            if conditioning_type != "motion_track":
                raise HTTPError(400, "Motion Track IC-LoRA only supports motion_track conditioning")
            lora_path = resolve_model_path(self.models_dir, self.config.model_download_specs, "ic_lora_motion_track")
            if not lora_path.exists():
                raise HTTPError(400, f"IC-LoRA model not found: {lora_path}")
            return lora_path, None, None, None

        if conditioning_type not in _UNION_CONDITIONING_TYPES:
            raise HTTPError(400, f"Unsupported conditioning_type for union IC-LoRA: {conditioning_type}")

        lora_path = resolve_model_path(self.models_dir, self.config.model_download_specs, "ic_lora")
        if not lora_path.exists():
            raise HTTPError(400, f"IC-LoRA model not found: {lora_path}")

        depth_model_path: Path | None = None
        pose_model_path: Path | None = None
        person_detector_model_path: Path | None = None

        if conditioning_type == "depth":
            depth_model_path = resolve_model_path(self.models_dir, self.config.model_download_specs, "depth_processor")
            if not depth_model_path.exists():
                raise HTTPError(400, f"Depth processor model not found: {depth_model_path}")
        elif conditioning_type == "pose":
            pose_model_path = resolve_model_path(self.models_dir, self.config.model_download_specs, "pose_processor")
            person_detector_model_path = resolve_model_path(
                self.models_dir,
                self.config.model_download_specs,
                "person_detector",
            )
            if not pose_model_path.exists():
                raise HTTPError(400, f"Pose processor model not found: {pose_model_path}")
            if not person_detector_model_path.exists():
                raise HTTPError(400, f"Person detector model not found: {person_detector_model_path}")

        return lora_path, depth_model_path, pose_model_path, person_detector_model_path

    def extract_conditioning(self, req: IcLoraExtractRequest) -> IcLoraExtractResponse:
        video_file = Path(req.video_path)
        if not video_file.exists():
            raise HTTPError(400, f"Video not found: {req.video_path}")

        cap = self._video_processor.open_video(str(video_file))
        info = self._video_processor.get_video_info(cap)
        target_frame = int(req.frame_time * float(info["fps"]))
        frame = self._video_processor.read_frame(cap, frame_idx=target_frame)
        self._video_processor.release(cap)

        if frame is None:
            raise HTTPError(400, "Could not read frame from video")

        ic_state: ICLoraState | None = None
        if req.conditioning_type in {"depth", "pose"}:
            lora_path, depth_model_path, pose_model_path, person_detector_model_path = self._resolve_ic_lora_resources(
                model_type=req.model_type,
                conditioning_type=req.conditioning_type,
            )
            ic_state = self._pipelines.load_ic_lora(
                str(lora_path),
                str(depth_model_path) if depth_model_path is not None else None,
                str(pose_model_path) if pose_model_path is not None else None,
                str(person_detector_model_path) if person_detector_model_path is not None else None,
            )

        result = self._build_conditioning_frame(frame, req.conditioning_type, ic_state)

        conditioning = self._video_processor.encode_frame_jpeg(result, quality=85)
        original = self._video_processor.encode_frame_jpeg(frame, quality=85)

        return IcLoraExtractResponse(
            conditioning="data:image/jpeg;base64," + base64.b64encode(conditioning).decode("utf-8"),
            original="data:image/jpeg;base64," + base64.b64encode(original).decode("utf-8"),
            conditioning_type=req.conditioning_type,
            model_type=req.model_type,
            frame_time=req.frame_time,
        )

    def _resolve_seed(self) -> int:
        settings = self.state.app_settings
        if settings.seed_locked:
            return settings.locked_seed
        return int(time.time()) % 2147483647

    def _get_total_vram_gb(self) -> int:
        if not torch.cuda.is_available():
            return 0
        props = cast(Any, torch.cuda.get_device_properties(0))  # pyright: ignore[reportUnknownMemberType]
        total_bytes = int(props.total_memory)
        return int((total_bytes + (1024**3 - 1)) // (1024**3))

    @staticmethod
    def _resolve_target_resolution(
        resolution: str,
        aspect_ratio: str,
    ) -> tuple[int, int]:
        landscape_sizes = {
            "540p": (896, 512),
            "720p": (1280, 768),
            "1080p": (1792, 1024),
        }
        base_width, base_height = landscape_sizes.get(resolution, landscape_sizes["540p"])
        if aspect_ratio == "9:16":
            return base_height, base_width
        return base_width, base_height

    def _select_generation_profile(
        self,
        *,
        resolution: str,
        aspect_ratio: str,
        fps: float,
    ) -> tuple[int, int, int]:
        manager = VRAMManager(
            self.config.device,
            self._get_total_vram_gb(),
            user_blocks_on_gpu=self.state.app_settings.num_blocks_to_swap,
            user_run_mode=self.state.app_settings.run_mode,
        )
        width, height = self._resolve_target_resolution(
            resolution,
            aspect_ratio,
        )
        max_frames = manager.get_max_frames(width, height, max(1, int(round(fps))))
        return width, height, max_frames

    def generate(self, req: IcLoraGenerateRequest) -> IcLoraGenerateResponse:
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        video_path = Path(req.video_path)
        if not video_path.exists():
            raise HTTPError(400, f"Video not found: {req.video_path}")
        lora_path, depth_model_path, pose_model_path, person_detector_model_path = self._resolve_ic_lora_resources(
            model_type=req.model_type,
            conditioning_type=req.conditioning_type,
        )

        generation_id = uuid.uuid4().hex[:8]
        t_total_start = time.perf_counter()
        logger.info("[ic-lora] Generation started (conditioning=%s)", req.conditioning_type)

        try:
            t_load_start = time.perf_counter()
            ic_state = self._pipelines.load_ic_lora(
                str(lora_path),
                str(depth_model_path) if depth_model_path is not None else None,
                str(pose_model_path) if pose_model_path is not None else None,
                str(person_detector_model_path) if person_detector_model_path is not None else None,
            )
            t_load_end = time.perf_counter()
            logger.info("[ic-lora] Pipeline load: %.2fs", t_load_end - t_load_start)

            self._generation.start_generation(generation_id)
            self._generation.update_progress("loading_model", 5, 0, 1)

            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(req.prompt, enhance_prompt=False)
            t_text_end = time.perf_counter()
            logger.info("[ic-lora] Text encoding (local): %.2fs", t_text_end - t_text_start)

            cap = self._video_processor.open_video(str(video_path))
            if not cap.isOpened():
                raise HTTPError(400, f"Cannot open video: {video_path}")
            info = self._video_processor.get_video_info(cap)
            input_width = int(info["width"])
            input_height = int(info["height"])
            source_frame_count = int(info["frame_count"])
            source_fps = float(info["fps"])
            requested_frame_count = source_frame_count
            if req.duration is not None:
                if req.duration <= 0:
                    raise HTTPError(400, "duration must be greater than 0 when provided")
                requested_from_duration = max(1, int(round(req.duration * source_fps)))
                requested_frame_count = min(source_frame_count, snap_frames_to_8k_plus_1(requested_from_duration))
            width, height, supported_frames = self._select_generation_profile(
                resolution=req.resolution,
                aspect_ratio=req.aspect_ratio,
                fps=source_fps,
            )

            cache_key = ConditioningCacheKey(
                str(video_path),
                req.conditioning_type,
                width,
                height,
                supported_frames,
                requested_frame_count,
            )
            cached = ic_state.conditioning_cache.get(cache_key)

            t_preprocess_start = 0.0
            t_preprocess_end = 0.0

            if req.conditioning_type == "motion_track" and cached is not None:
                self._video_processor.release(cap)
                control_video_path = cached.control_video_path
                frame_count = cached.frame_count
                fps = cached.fps
                logger.info("[ic-lora] Motion-track overlay cache hit for %s", video_path.name)
            elif req.conditioning_type == "motion_track":
                self._video_processor.release(cap)
                t_preprocess_start = time.perf_counter()
                control_video_path, frame_count, fps = create_motion_track_overlay_video(
                    video_path=str(video_path),
                    output_dir=self.config.outputs_dir / "_normalized_inputs",
                )
                t_preprocess_end = time.perf_counter()
                ic_state.conditioning_cache.put(
                    cache_key, ConditioningCacheEntry(control_video_path, frame_count, fps)
                )
                logger.info(
                    "[ic-lora] Built motion-track overlay control %s -> %s (%d frames, %.2fs)",
                    video_path.name,
                    control_video_path,
                    frame_count,
                    t_preprocess_end - t_preprocess_start,
                )
            elif cached is not None:
                self._video_processor.release(cap)
                control_video_path = cached.control_video_path
                frame_count = cached.frame_count
                fps = cached.fps
                logger.info("[ic-lora] Conditioning cache hit for %s/%s", video_path.name, req.conditioning_type)
            else:
                t_preprocess_start = time.perf_counter()

                frame_count = requested_frame_count
                fps = source_fps

                control_video_path = str(
                    self.config.outputs_dir / f"_control_{req.conditioning_type}_{uuid.uuid4().hex[:8]}.mp4"
                )
                writer = self._video_processor.create_writer(
                    control_video_path,
                    fourcc="mp4v",
                    fps=fps,
                    size=(width, height),
                )

                frame_idx = 0
                while frame_idx < frame_count:
                    frame = self._video_processor.read_frame(cap)
                    if frame is None:
                        break
                    resized_frame = frame
                    if int(info["width"]) != width or int(info["height"]) != height:
                        resized_frame = self._video_processor.resize_frame(frame, (width, height))
                    control_frame = self._build_conditioning_frame(resized_frame, req.conditioning_type, ic_state)
                    writer.write(control_frame)
                    frame_idx += 1
                    if frame_count > 0 and frame_idx % 8 == 0:
                        preprocess_progress = 10 + math.floor((frame_idx / frame_count) * 35)
                        self._generation.update_progress("preprocessing", preprocess_progress, frame_idx, frame_count)

                self._video_processor.release(cap)
                self._video_processor.release(writer)
                t_preprocess_end = time.perf_counter()
                logger.info(
                    "[ic-lora] Preprocessing (%s, %d frames): %.2fs",
                    req.conditioning_type, frame_idx, t_preprocess_end - t_preprocess_start,
                )

                ic_state.conditioning_cache.put(
                    cache_key, ConditioningCacheEntry(control_video_path, frame_count, fps)
                )

            if frame_count > supported_frames or (frame_count - 1) % 8 != 0:
                normalized = downsample_video_temporally_for_ltx(
                    video_path=control_video_path,
                    output_dir=self.config.outputs_dir / "_normalized_inputs",
                    target_max_frames=supported_frames,
                )
                control_video_path = normalized.path
                frame_count = normalized.normalized_frames
                fps = normalized.fps
                logger.info(
                    "[ic-lora] Low-VRAM control normalization %s -> %s (frames %d->%d fps %.2f->%.2f)",
                    video_path,
                    control_video_path,
                    normalized.original_frames,
                    normalized.normalized_frames,
                    source_fps,
                    fps,
                )
                ic_state.conditioning_cache.put(
                    cache_key, ConditioningCacheEntry(control_video_path, frame_count, fps)
                )

            images: list[ImageConditioningInput] = [
                ImageConditioningInput(path=img.path, frame_idx=int(img.frame), strength=float(img.strength))
                for img in req.images
            ]

            self._generation.update_progress("inference", 50, 0, 1)

            output_path = (
                self.config.outputs_dir / f"ic_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
            )

            t_inference_start = time.perf_counter()
            skip_stage_2 = False
            try:
                ic_state.pipeline.generate(
                    prompt=req.prompt,
                    seed=self._resolve_seed(),
                    height=height,
                    width=width,
                    num_frames=frame_count,
                    frame_rate=fps,
                    images=images,
                    video_conditioning=[(control_video_path, req.conditioning_strength)],
                    output_path=str(output_path),
                    source_audio_path=str(video_path),
                    source_audio_max_duration=frame_count / fps if fps > 0 else None,
                    skip_stage_2=skip_stage_2,
                )
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._generation.update_progress("retrying_low_vram", 60, 0, 1)
                skip_stage_2 = True
                logger.warning("[ic-lora] Retrying generation with skip_stage_2 due to OOM")
                ic_state.pipeline.generate(
                    prompt=req.prompt,
                    seed=self._resolve_seed(),
                    height=height,
                    width=width,
                    num_frames=frame_count,
                    frame_rate=fps,
                    images=images,
                    video_conditioning=[(control_video_path, req.conditioning_strength)],
                    output_path=str(output_path),
                    source_audio_path=str(video_path),
                    source_audio_max_duration=frame_count / fps if fps > 0 else None,
                    skip_stage_2=skip_stage_2,
                )
            t_inference_end = time.perf_counter()
            logger.info("[ic-lora] Inference: %.2fs", t_inference_end - t_inference_start)

            t_total_end = time.perf_counter()
            preprocess_time = (t_preprocess_end - t_preprocess_start) if cached is None else 0.0
            logger.info(
                "[ic-lora] Total generation: %.2fs (load=%.2fs, text=%.2fs, preprocess=%.2fs, inference=%.2fs)",
                t_total_end - t_total_start,
                t_load_end - t_load_start,
                t_text_end - t_text_start,
                preprocess_time,
                t_inference_end - t_inference_start,
            )

            self._generation.update_progress("complete", 100, 1, 1)
            self._generation.complete_generation(str(output_path))
            return IcLoraGenerateResponse(status="complete", video_path=str(output_path))

        except HTTPError:
            self._generation.fail_generation("IC-LoRA generation failed")
            raise
        except Exception as exc:
            self._generation.fail_generation(str(exc))
            if "cancelled" in str(exc).lower():
                return IcLoraGenerateResponse(status="cancelled")
            raise HTTPError(500, f"Generation error: {exc}") from exc
        finally:
            self._text.clear_api_embeddings()
