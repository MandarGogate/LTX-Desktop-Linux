from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np


def create_motion_track_overlay_video(
    *,
    video_path: str,
    output_dir: Path,
    max_points: int = 32,
    history: int = 12,
) -> tuple[str, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError("Failed to read first frame for motion-track conversion")

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=7,
    )
    if points is None or len(points) == 0:
        # fallback grid
        grid_points: list[list[float]] = []
        for y in np.linspace(height * 0.2, height * 0.8, 4):
            for x in np.linspace(width * 0.2, width * 0.8, 4):
                grid_points.append([[float(x), float(y)]])
        points = np.array(grid_points[:max_points], dtype=np.float32)

    colors = [
        (255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 220, 80),
        (255, 80, 220), (80, 255, 220), (220, 120, 255), (255, 140, 80),
    ]
    track_histories: list[list[tuple[int, int]]] = []
    for p in points.reshape(-1, 2):
        track_histories.append([(int(p[0]), int(p[1]))])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"motion_track_overlay_{uuid.uuid4().hex[:8]}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0
    current_points = points
    current_frame = first_frame

    def render(frame: np.ndarray, histories: list[list[tuple[int, int]]]) -> np.ndarray:
        canvas = (frame.astype(np.float32) * 0.45).astype(np.uint8)
        for idx, hist in enumerate(histories):
            color = colors[idx % len(colors)]
            for i in range(1, len(hist)):
                cv2.line(canvas, hist[i - 1], hist[i], color, 2, lineType=cv2.LINE_AA)
            if hist:
                cv2.circle(canvas, hist[-1], 4, color, thickness=-1, lineType=cv2.LINE_AA)
        return canvas

    while True:
        overlay = render(current_frame, track_histories)
        writer.write(overlay)
        frame_count += 1

        ok, next_frame = cap.read()
        if not ok or next_frame is None:
            break

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            next_gray,
            current_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        if next_points is None or status is None:
            next_points = current_points
            status = np.ones((len(track_histories), 1), dtype=np.uint8)

        for idx, (pt, st) in enumerate(zip(next_points.reshape(-1, 2), status.reshape(-1))):
            if idx >= len(track_histories):
                break
            if st:
                x = int(np.clip(pt[0], 0, width - 1))
                y = int(np.clip(pt[1], 0, height - 1))
                track_histories[idx].append((x, y))
                if len(track_histories[idx]) > history:
                    track_histories[idx] = track_histories[idx][-history:]

        current_points = next_points.reshape(-1, 1, 2).astype(np.float32)
        prev_gray = next_gray
        current_frame = next_frame

    cap.release()
    writer.release()
    return str(output_path), frame_count, fps
