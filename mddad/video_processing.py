from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    readable: bool
    fps: float = 0.0
    frame_count: int = 0
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    error: str | None = None


def extract_video_metadata(video_path):
    path = Path(video_path)
    if not path.exists():
        return VideoMetadata(readable=False, error="Video file is missing.")
    if path.stat().st_size == 0:
        return VideoMetadata(readable=False, error="Video file is empty.")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return VideoMetadata(readable=False, error="OpenCV could not open the video file.")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if frame_count <= 0:
            return VideoMetadata(readable=False, error="Video contains zero readable frames.")

        first_frame_ok, _frame = capture.read()
        if not first_frame_ok:
            return VideoMetadata(readable=False, error="OpenCV could not read frames from the video.")

        duration = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(
            readable=True,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            width=width,
            height=height,
        )
    except cv2.error as exc:
        return VideoMetadata(readable=False, error=f"OpenCV error: {exc}")
    finally:
        capture.release()


def iter_sampled_frames(video_path, sample_fps=5.0, every_n_frames=None):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open the video file.")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_interval = every_n_frames
        if frame_interval is None:
            if source_fps > 0 and sample_fps > 0:
                frame_interval = max(int(round(source_fps / sample_fps)), 1)
            else:
                frame_interval = 1

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_interval == 0:
                timestamp_seconds = frame_index / source_fps if source_fps > 0 else 0.0
                yield frame, frame_index, timestamp_seconds

            frame_index += 1
    finally:
        capture.release()


def count_sampled_frames(video_path, sample_fps=5.0, every_n_frames=None):
    count = 0
    for _frame, _frame_index, _timestamp_seconds in iter_sampled_frames(
        video_path,
        sample_fps=sample_fps,
        every_n_frames=every_n_frames,
    ):
        count += 1
    return count
