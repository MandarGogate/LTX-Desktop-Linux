from __future__ import annotations

import math
import shutil
import subprocess
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
