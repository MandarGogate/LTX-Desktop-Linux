from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NormalizedVideoInfo:
    path: str
    changed: bool
    original_frames: int
    normalized_frames: int
    original_width: int
    original_height: int
    normalized_width: int
    normalized_height: int
    fps: float


@dataclass(frozen=True)
class TemporalChunk:
    index: int
    start_frame: int
    frame_count: int
    keep_start_frame: int
    keep_frame_count: int


def snap_frames_to_8k_plus_1(num_frames: int) -> int:
    if num_frames <= 1:
        return 1
    return ((num_frames - 1) // 8) * 8 + 1


def crop_down_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        return multiple
    cropped = (value // multiple) * multiple
    return cropped if cropped > 0 else multiple


def _run_ffmpeg(command: list[str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    result = subprocess.run([ffmpeg, *command], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")


def normalize_video_for_ltx(
    *,
    video_path: str,
    output_dir: Path,
    fps: float,
    num_frames: int,
    width: int,
    height: int,
    keep_audio: bool,
    target_width: int | None = None,
    target_height: int | None = None,
) -> NormalizedVideoInfo:
    normalized_frames = snap_frames_to_8k_plus_1(num_frames)
    normalized_width = target_width if target_width is not None else crop_down_to_multiple(width, 32)
    normalized_height = target_height if target_height is not None else crop_down_to_multiple(height, 32)

    changed = (
        normalized_frames != num_frames
        or normalized_width != width
        or normalized_height != height
    )
    if not changed:
        return NormalizedVideoInfo(
            path=video_path,
            changed=False,
            original_frames=num_frames,
            normalized_frames=num_frames,
            original_width=width,
            original_height=height,
            normalized_width=width,
            normalized_height=height,
            fps=fps,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ltx_norm_{uuid.uuid4().hex[:8]}.mp4"

    vf_parts: list[str] = []
    if target_width is not None and target_height is not None:
        vf_parts.append(
            f"scale={normalized_width}:{normalized_height}:force_original_aspect_ratio=increase"
        )
        vf_parts.append(f"crop={normalized_width}:{normalized_height}")
    elif normalized_width != width or normalized_height != height:
        crop_x = max(0, (width - normalized_width) // 2)
        crop_y = max(0, (height - normalized_height) // 2)
        vf_parts.append(
            f"crop={normalized_width}:{normalized_height}:{crop_x}:{crop_y}"
        )

    command = [
        "-y",
        "-i",
        video_path,
    ]
    if vf_parts:
        command.extend(["-vf", ",".join(vf_parts)])
    command.extend([
        "-frames:v",
        str(normalized_frames),
        "-c:v",
        "mpeg4",
        "-q:v",
        "3",
        "-pix_fmt",
        "yuv420p",
    ])
    if keep_audio:
        command.extend(["-c:a", "aac"])
    else:
        command.append("-an")
    command.append(str(output_path))

    _run_ffmpeg(command)

    return NormalizedVideoInfo(
        path=str(output_path),
        changed=True,
        original_frames=num_frames,
        normalized_frames=normalized_frames,
        original_width=width,
        original_height=height,
        normalized_width=normalized_width,
        normalized_height=normalized_height,
        fps=fps,
    )


def plan_temporal_chunks(
    *,
    total_frames: int,
    max_chunk_frames: int,
    overlap_frames: int,
) -> list[TemporalChunk]:
    chunk_frames = snap_frames_to_8k_plus_1(max_chunk_frames)
    if total_frames <= chunk_frames:
        return [
            TemporalChunk(
                index=0,
                start_frame=0,
                frame_count=total_frames,
                keep_start_frame=0,
                keep_frame_count=total_frames,
            )
        ]

    safe_overlap = max(1, min(overlap_frames, chunk_frames - 1))
    step = max(1, chunk_frames - safe_overlap)
    starts = [0]
    while True:
        last = starts[-1]
        if last + chunk_frames >= total_frames:
            break
        next_start = last + step
        final_start = max(0, total_frames - chunk_frames)
        if next_start >= final_start:
            if final_start > last:
                starts.append(final_start)
            break
        starts.append(next_start)

    boundaries: list[int] = [0]
    for idx in range(len(starts) - 1):
        current_start = starts[idx]
        next_start = starts[idx + 1]
        current_end = min(total_frames, current_start + chunk_frames)
        next_end = min(total_frames, next_start + chunk_frames)
        overlap_start = next_start
        overlap_end = min(current_end, next_end)
        if overlap_end <= overlap_start:
            boundaries.append(next_start)
            continue
        boundaries.append(overlap_start + ((overlap_end - overlap_start) // 2))
    boundaries.append(total_frames)

    chunks: list[TemporalChunk] = []
    for idx, start in enumerate(starts):
        keep_global_start = boundaries[idx]
        keep_global_end = boundaries[idx + 1]
        chunks.append(
            TemporalChunk(
                index=idx,
                start_frame=start,
                frame_count=min(chunk_frames, total_frames - start),
                keep_start_frame=max(0, keep_global_start - start),
                keep_frame_count=max(0, keep_global_end - keep_global_start),
            )
        )
    return chunks


def extract_video_frame_range(
    *,
    video_path: str,
    output_path: str,
    start_frame: int,
    frame_count: int,
) -> None:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Failed to read video dimensions: {video_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    written = 0
    frame_idx = 0
    try:
        while written < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= start_frame:
                writer.write(frame)
                written += 1
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    if written != frame_count:
        raise RuntimeError(
            f"Video slice produced {written} frame(s), expected {frame_count}: {video_path}"
        )


def extract_video_frame_to_image(
    *,
    video_path: str,
    output_path: str,
    frame_idx: int,
) -> None:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to extract frame {frame_idx} from video: {video_path}"
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(output_path, frame):
            raise RuntimeError(f"Failed to write frame image: {output_path}")
    finally:
        cap.release()


def trim_video_for_frame_range(
    *,
    video_path: str,
    output_path: str,
    fps: float,
    start_frame: int,
    frame_count: int,
) -> None:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 24.0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Failed to read video dimensions: {video_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter.fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )

    written = 0
    frame_idx = 0
    try:
        while written < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= start_frame:
                writer.write(frame)
                written += 1
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    if written != frame_count:
        raise RuntimeError(
            f"Trimmed video produced {written} frame(s), expected {frame_count}: {video_path}"
        )


def concat_videos(
    *,
    input_paths: list[str],
    output_path: str,
) -> None:
    if not input_paths:
        raise RuntimeError("No input videos provided for concatenation")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as manifest:
        manifest_path = Path(manifest.name)
        for path in input_paths:
            manifest.write(f"file '{path}'\n")

    try:
        # The trimmed chunks are encoded with a consistent codec/format, so we
        # can concatenate via stream copy and avoid one more lossy re-encode as
        # well as the huge files produced by MPEG-4 qscale output.
        _run_ffmpeg(
            [
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
    finally:
        manifest_path.unlink(missing_ok=True)


def mux_video_with_audio(
    *,
    video_path: str,
    audio_source_path: str,
    output_path: str,
    audio_start_time: float = 0.0,
    audio_duration: float | None = None,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "-y",
        "-i",
        video_path,
    ]
    if audio_start_time > 0:
        command.extend(["-ss", f"{audio_start_time:.6f}"])
    if audio_duration is not None:
        command.extend(["-t", f"{audio_duration:.6f}"])
    command.extend(
        [
            "-i",
            audio_source_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-shortest",
            output_path,
        ]
    )
    _run_ffmpeg(command)


def downsample_video_temporally_for_ltx(
    *,
    video_path: str,
    output_dir: Path,
    target_max_frames: int,
) -> NormalizedVideoInfo:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if num_frames <= target_max_frames and (num_frames - 1) % 8 == 0:
        cap.release()
        return NormalizedVideoInfo(
            path=video_path,
            changed=False,
            original_frames=num_frames,
            normalized_frames=num_frames,
            original_width=width,
            original_height=height,
            normalized_width=width,
            normalized_height=height,
            fps=fps,
        )

    stride = max(1, math.ceil((num_frames - 1) / max(1, target_max_frames - 1)))
    sampled_frames = ((num_frames - 1) // stride) + 1
    normalized_frames = snap_frames_to_8k_plus_1(sampled_frames)
    normalized_width = crop_down_to_multiple(width, 32)
    normalized_height = crop_down_to_multiple(height, 32)
    output_fps = fps / stride if stride > 1 else fps

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ltx_temporal_{uuid.uuid4().hex[:8]}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        output_fps,
        (normalized_width, normalized_height),
    )

    written = 0
    frame_index = 0
    try:
        while written < normalized_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                if normalized_width != width or normalized_height != height:
                    crop_x = max(0, (width - normalized_width) // 2)
                    crop_y = max(0, (height - normalized_height) // 2)
                    frame = frame[
                        crop_y:crop_y + normalized_height,
                        crop_x:crop_x + normalized_width,
                    ]
                writer.write(frame)
                written += 1
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if written <= 0:
        raise RuntimeError("Temporal downsampling produced no frames")

    return NormalizedVideoInfo(
        path=str(output_path),
        changed=True,
        original_frames=num_frames,
        normalized_frames=written,
        original_width=width,
        original_height=height,
        normalized_width=normalized_width,
        normalized_height=normalized_height,
        fps=output_fps,
    )
