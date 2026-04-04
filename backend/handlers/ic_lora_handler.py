"""IC-LoRA endpoints orchestration handler."""

from __future__ import annotations

import base64
import gc
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
from server_utils.ltx_video_normalization import (
    concat_videos,
    extract_video_frame_range,
    extract_video_frame_to_image,
    mux_video_with_audio,
    plan_temporal_chunks,
    snap_frames_to_8k_plus_1,
    trim_video_for_frame_range,
)
from server_utils.motion_track_overlay import create_motion_track_overlay_video
from state.app_state_types import AppState, ICLoraState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


_UNION_CONDITIONING_TYPES = {"canny", "depth", "pose"}


def _chunk_image_inputs(
    images: list[ImageConditioningInput],
    *,
    chunk_start_frame: int,
    chunk_frame_count: int,
) -> list[ImageConditioningInput]:
    if chunk_frame_count <= 0:
        return []

    chunk_end_frame = chunk_start_frame + chunk_frame_count
    localized: list[ImageConditioningInput] = []
    for img in images:
        if img.frame_idx < chunk_start_frame:
            localized.append(
                ImageConditioningInput(
                    path=img.path,
                    frame_idx=0,
                    strength=img.strength,
                )
            )
        elif img.frame_idx < chunk_end_frame:
            localized.append(
                ImageConditioningInput(
                    path=img.path,
                    frame_idx=img.frame_idx - chunk_start_frame,
                    strength=img.strength,
                )
            )

    return localized


def _ic_lora_inference_progress(
    *,
    phase: str,
    current_step: int | None,
    total_steps: int | None,
    skip_stage_2: bool,
) -> tuple[int, int | None, int | None]:
    if total_steps is None or total_steps <= 0 or current_step is None:
        match phase:
            case "denoising_stage_1":
                return (50, current_step, total_steps)
            case "denoising_stage_2":
                return (75 if not skip_stage_2 else 95, current_step, total_steps)
            case _:
                return (50, current_step, total_steps)

    clamped_step = max(0, min(current_step, total_steps))
    if phase == "denoising_stage_1":
        start = 50
        end = 95 if skip_stage_2 else 75
    elif phase == "denoising_stage_2":
        start = 75
        end = 95
    else:
        start = 50
        end = 95

    progress = start + math.floor(((end - start) * clamped_step) / max(total_steps, 1))
    return (progress, clamped_step, total_steps)


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

        # IC-LoRA is much less memory-stable than the main video pipelines.
        # The generic VRAM estimator is too optimistic here and can keep long
        # clips on the non-chunked path, which then OOMs at 720p/1080p.
        # Use conservative per-resolution caps so longer clips chunk earlier.
        resolution_cap = 161
        if max(width, height) >= 1792:
            resolution_cap = 81
        elif max(width, height) >= 1280:
            resolution_cap = 113

        max_frames = min(max_frames, resolution_cap)
        return width, height, max_frames

    def _run_ic_lora_generate_with_retry(
        self,
        *,
        ic_state: ICLoraState,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        control_video_path: str,
        conditioning_strength: float,
        output_path: str,
        source_audio_path: str,
        source_audio_start_time: float = 0.0,
        source_audio_max_duration: float | None = None,
        progress_callback: Any = None,
    ) -> None:
        skip_stage_2 = False
        should_retry_low_vram = False
        try:
            ic_state.pipeline.generate(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                video_conditioning=[(control_video_path, conditioning_strength)],
                output_path=output_path,
                source_audio_path=source_audio_path,
                source_audio_start_time=source_audio_start_time,
                source_audio_max_duration=source_audio_max_duration,
                skip_stage_2=skip_stage_2,
                progress_callback=progress_callback,
            )
            return
        except torch.OutOfMemoryError:
            logger.warning("[ic-lora] Retrying generation with skip_stage_2 due to OOM")
            should_retry_low_vram = True

        if not should_retry_low_vram:
            return

        # Important: retry outside the except block. While the exception is
        # active, its traceback can keep large tensors/modules alive and make a
        # low-VRAM retry fail immediately with almost no free CUDA memory.
        skip_stage_2 = True
        self._generation.update_progress("retrying_low_vram", 60, 0, 1)
        gc.collect()
        if hasattr(ic_state.pipeline, "_cleanup_generation_state"):
            try:
                getattr(ic_state.pipeline, "_cleanup_generation_state")()
            except Exception:
                logger.debug("[ic-lora] Pipeline retry cleanup failed", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                try:
                    ipc_collect()
                except Exception:
                    logger.debug("[ic-lora] torch.cuda.ipc_collect failed", exc_info=True)

        ic_state.pipeline.generate(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            video_conditioning=[(control_video_path, conditioning_strength)],
            output_path=output_path,
            source_audio_path=source_audio_path,
            source_audio_start_time=source_audio_start_time,
            source_audio_max_duration=source_audio_max_duration,
            skip_stage_2=skip_stage_2,
            progress_callback=progress_callback,
        )

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
            # IC-LoRA later calls encode_text(None, ...), expecting the global
            # text-encoder patch to serve a cached local encoder. Prime that
            # cache now by forcing the pipeline's stage-1 model ledger to build
            # its text encoder once.
            stage_1_model_ledger = getattr(getattr(ic_state.pipeline, "pipeline", None), "stage_1_model_ledger", None)
            if stage_1_model_ledger is not None and hasattr(stage_1_model_ledger, "text_encoder"):
                try:
                    stage_1_model_ledger.text_encoder()
                except Exception:
                    logger.debug("[ic-lora] Failed to prewarm cached text encoder", exc_info=True)
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

                # Snap frame count to 8k+1 before writing the control video
                # so no post-hoc normalization is needed.
                frame_count = snap_frames_to_8k_plus_1(requested_frame_count)
                if frame_count < 9:
                    frame_count = 9
                frame_count = min(frame_count, requested_frame_count)
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

                # Update frame_count to what was actually written
                frame_count = frame_idx
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

            use_chunked_generation = frame_count > supported_frames and fps > 0

            images: list[ImageConditioningInput] = [
                ImageConditioningInput(path=img.path, frame_idx=int(img.frame), strength=float(img.strength))
                for img in req.images
            ]

            self._generation.update_progress("inference", 50, 0, 1)

            seed = self._resolve_seed()

            output_path = (
                self.config.outputs_dir / f"ic_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
            )

            t_inference_start = time.perf_counter()
            skip_stage_2 = False

            def _progress_callback(phase: str, current_step: int | None, total_steps: int | None) -> None:
                progress, effective_step, effective_total = _ic_lora_inference_progress(
                    phase=phase,
                    current_step=current_step,
                    total_steps=total_steps,
                    skip_stage_2=skip_stage_2,
                )
                self._generation.update_progress(phase, progress, effective_step, effective_total)

            if use_chunked_generation:
                overlap_frames = 17 if supported_frames > 33 else max(1, supported_frames // 4)
                chunk_frame_budget = supported_frames
                if max(width, height) >= 1792:
                    chunk_frame_budget = min(chunk_frame_budget, 113)
                chunks = plan_temporal_chunks(
                    total_frames=frame_count,
                    max_chunk_frames=chunk_frame_budget,
                    overlap_frames=overlap_frames,
                )
                logger.info(
                    "[ic-lora] Using temporal chunking: %d chunks (frames=%d chunk=%d overlap=%d fps=%.2f)",
                    len(chunks),
                    frame_count,
                    chunks[0].frame_count,
                    overlap_frames,
                    fps,
                )
                chunk_control_paths: list[str] = []
                for chunk in chunks:
                    chunk_control_path = str(
                        self.config.outputs_dir / "_normalized_inputs" / f"ltx_ic_chunk_control_{uuid.uuid4().hex[:8]}.mp4"
                    )
                    extract_video_frame_range(
                        video_path=control_video_path,
                        output_path=chunk_control_path,
                        start_frame=chunk.start_frame,
                        frame_count=chunk.frame_count,
                    )
                    chunk_control_paths.append(chunk_control_path)
                chunk_output_paths: list[str] = []
                trimmed_chunk_paths: list[str] = []
                previous_chunk_bridge_image: str | None = None
                for chunk, chunk_control_path in zip(chunks, chunk_control_paths, strict=True):
                    chunk_output_path = str(
                        self.config.outputs_dir / f"ic_lora_chunk_{uuid.uuid4().hex[:8]}.mp4"
                    )
                    local_images = _chunk_image_inputs(
                        images,
                        chunk_start_frame=chunk.start_frame,
                        chunk_frame_count=chunk.frame_count,
                    )
                    if previous_chunk_bridge_image is not None:
                        local_images = [
                            ImageConditioningInput(
                                path=previous_chunk_bridge_image,
                                frame_idx=0,
                                strength=1.0,
                            ),
                            *local_images,
                        ]
                    logger.info(
                        "[ic-lora] Chunk %d/%d: start=%d frames=%d keep=%d+%d",
                        chunk.index + 1,
                        len(chunks),
                        chunk.start_frame,
                        chunk.frame_count,
                        chunk.keep_start_frame,
                        chunk.keep_frame_count,
                    )
                    self._run_ic_lora_generate_with_retry(
                        ic_state=ic_state,
                        prompt=req.prompt,
                        seed=seed,
                        height=height,
                        width=width,
                        num_frames=chunk.frame_count,
                        frame_rate=fps,
                        images=local_images,
                        control_video_path=chunk_control_path,
                        conditioning_strength=req.conditioning_strength,
                        output_path=chunk_output_path,
                        source_audio_path=str(video_path),
                        source_audio_start_time=chunk.start_frame / fps,
                        source_audio_max_duration=chunk.frame_count / fps,
                        progress_callback=_progress_callback,
                    )
                    chunk_output_paths.append(chunk_output_path)
                    bridge_frame_idx = max(
                        0,
                        min(
                            chunk.frame_count - 1,
                            chunk.keep_start_frame + chunk.keep_frame_count - 1,
                        ),
                    )
                    previous_chunk_bridge_image = str(
                        self.config.outputs_dir / "_normalized_inputs" / f"ltx_ic_chunk_bridge_{uuid.uuid4().hex[:8]}.png"
                    )
                    extract_video_frame_to_image(
                        video_path=chunk_output_path,
                        output_path=previous_chunk_bridge_image,
                        frame_idx=bridge_frame_idx,
                    )
                    trimmed_chunk_path = str(
                        self.config.outputs_dir / f"ic_lora_chunk_trim_{uuid.uuid4().hex[:8]}.mp4"
                    )
                    trim_video_for_frame_range(
                        video_path=chunk_output_path,
                        output_path=trimmed_chunk_path,
                        fps=fps,
                        start_frame=chunk.keep_start_frame,
                        frame_count=chunk.keep_frame_count,
                    )
                    trimmed_chunk_paths.append(trimmed_chunk_path)
                    if chunk.index + 1 < len(chunks):
                        ic_state = self._pipelines.reload_ic_lora_during_generation(
                            str(lora_path),
                            str(depth_model_path) if depth_model_path is not None else None,
                            str(pose_model_path) if pose_model_path is not None else None,
                            str(person_detector_model_path) if person_detector_model_path is not None else None,
                        )
                concat_output_path = str(
                    self.config.outputs_dir / f"ic_lora_concat_{uuid.uuid4().hex[:8]}.mp4"
                )
                concat_videos(input_paths=trimmed_chunk_paths, output_path=concat_output_path)
                mux_video_with_audio(
                    video_path=concat_output_path,
                    audio_source_path=str(video_path),
                    output_path=str(output_path),
                    audio_duration=frame_count / fps if fps > 0 else None,
                )
            else:
                self._run_ic_lora_generate_with_retry(
                    ic_state=ic_state,
                    prompt=req.prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=frame_count,
                    frame_rate=fps,
                    images=images,
                    control_video_path=control_video_path,
                    conditioning_strength=req.conditioning_strength,
                    output_path=str(output_path),
                    source_audio_path=str(video_path),
                    source_audio_max_duration=frame_count / fps if fps > 0 else None,
                    progress_callback=_progress_callback,
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
            self._pipelines.unload_gpu_pipeline()
            return IcLoraGenerateResponse(status="complete", video_path=str(output_path))

        except HTTPError:
            self._generation.fail_generation("IC-LoRA generation failed")
            self._pipelines.unload_gpu_pipeline()
            raise
        except Exception as exc:
            self._generation.fail_generation(str(exc))
            self._pipelines.unload_gpu_pipeline()
            if "cancelled" in str(exc).lower():
                return IcLoraGenerateResponse(status="cancelled")
            raise HTTPError(500, f"Generation error: {exc}") from exc
        finally:
            self._text.clear_api_embeddings()
