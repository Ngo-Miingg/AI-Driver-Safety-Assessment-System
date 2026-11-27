"""Capture verified driver actions with a rolling pre-buffer.

The tool uses the same DriverActionOpenVINORuntime as the product runtime.
It keeps recent camera frames in memory, then writes the preceding frames when
the temporal action model confirms a class. Press Q to stop the camera.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config  # noqa: E402
from mddad.driver_action_openvino import (  # noqa: E402
    DriverActionOpenVINORuntime,
    OPENVINO_LABELS,
)


@dataclass
class BufferedFrame:
    timestamp: float
    frame: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record OpenVINO driver actions with a temporal pre-buffer."
    )
    parser.add_argument("--camera", default="0", help="Camera index or video source path")
    parser.add_argument("--device", default="auto", help="OpenVINO device, e.g. CPU, GPU, GPU.1")
    parser.add_argument("--pre-buffer", type=float, default=3.0)
    parser.add_argument("--post-buffer", type=float, default=1.0)
    parser.add_argument("--confirm-confidence", type=float, default=0.75)
    parser.add_argument("--end-confidence", type=float, default=0.60)
    parser.add_argument("--end-delay", type=float, default=2.0)
    parser.add_argument("--output", default="outputs/action_captures")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def source_value(value: str):
    return int(value) if str(value).isdigit() else value


def safe_label(label: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_") or "action"


def build_runtime(args: argparse.Namespace):
    config = {name: getattr(Config, name) for name in dir(Config) if name.isupper()}
    config["DRIVER_ACTION_DEVICE"] = args.device
    # The runtime only needs app.config, so this avoids importing the Flask app
    # and avoids starting database, SocketIO, or webcam services twice.
    return DriverActionOpenVINORuntime(SimpleNamespace(config=config))


def write_event(output_dir: Path, index: int, label: str, frames: list[BufferedFrame],
                fps: float, metadata: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:03d}_{safe_label(label)}_{time.strftime('%Y%m%d_%H%M%S')}"
    first = frames[0].frame
    height, width = first.shape[:2]
    video_path = output_dir / f"{stem}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video writer: {video_path}")
    for item in frames:
        writer.write(item.frame)
    writer.release()
    metadata_path = output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return video_path


def draw_overlay(frame, observation, recording: bool, fps: float, event_label: str | None):
    output = frame.copy()
    label = observation.instant_label or "warming_up"
    confidence = observation.instant_confidence or 0.0
    accepted = observation.label or "not_confirmed"
    lines = [
        f"Action: {label} ({confidence:.3f})",
        f"Accepted: {accepted}",
        f"Model buffer: {observation.buffer_size}/{observation.buffer_target}",
        f"FPS: {fps:.1f} | Device: OpenVINO",
        "RECORDING EVENT" if recording else "Waiting for confirmation",
    ]
    if event_label:
        lines.append(f"Event: {event_label}")
    color = (0, 180, 255) if recording else (80, 220, 80)
    for row, text in enumerate(lines):
        y = 32 + row * 28
        cv2.putText(output, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
    return output


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    runtime = build_runtime(args)
    if not runtime.ready:
        print(f"OpenVINO action runtime failed: {runtime.load_error}", file=sys.stderr)
        return 2
    print(f"OpenVINO action runtime ready on {runtime.resolved_device}")
    print("Classes:", ", ".join(OPENVINO_LABELS))

    capture = cv2.VideoCapture(source_value(args.camera))
    if not capture.isOpened():
        print(f"Cannot open camera/source: {args.camera}", file=sys.stderr)
        return 3
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = source_fps if source_fps >= 5.0 else 20.0
    buffer: deque[BufferedFrame] = deque()
    recording_frames: list[BufferedFrame] = []
    recording = False
    event_label = None
    event_start = None
    confirmed_at = None
    below_since = None
    event_index = 0
    frame_count = 0
    last_tick = time.monotonic()
    measured_fps = fps

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.monotonic()
            frame_count += 1
            elapsed = max(now - last_tick, 1e-6)
            measured_fps = 0.9 * measured_fps + 0.1 / elapsed
            last_tick = now
            current = BufferedFrame(now, frame.copy())
            buffer.append(current)
            while buffer and now - buffer[0].timestamp > args.pre_buffer:
                buffer.popleft()

            observation = runtime.observe(frame)
            active_label = observation.label
            active_confidence = observation.confidence or 0.0
            confirmed = bool(
                active_label
                and active_confidence >= args.confirm_confidence
            )

            if not recording and confirmed:
                recording = True
                event_label = active_label
                event_start = buffer[0].timestamp if buffer else now
                confirmed_at = now
                below_since = None
                recording_frames = list(buffer)
                print(f"START {event_label} confidence={active_confidence:.3f}")
            elif recording:
                recording_frames.append(current)
                same_label = active_label == event_label
                if same_label and active_confidence >= args.end_confidence:
                    below_since = None
                elif below_since is None:
                    below_since = now

                ended = below_since is not None and now - below_since >= args.end_delay
                if ended:
                    tail_until = now + args.post_buffer
                    while time.monotonic() < tail_until:
                        ok_tail, tail = capture.read()
                        if not ok_tail:
                            break
                        tail_now = time.monotonic()
                        recording_frames.append(BufferedFrame(tail_now, tail.copy()))
                        if not args.no_preview:
                            preview = draw_overlay(tail, observation, True, measured_fps, event_label)
                            cv2.imshow("OpenVINO Action Capture", preview)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                tail_until = 0
                                break
                    event_index += 1
                    duration = recording_frames[-1].timestamp - recording_frames[0].timestamp
                    metadata = {
                        "label": event_label,
                        "duration_seconds": round(duration, 3),
                        "pre_buffer_seconds": args.pre_buffer,
                        "post_buffer_seconds": args.post_buffer,
                        "confirmed_confidence": round(float(active_confidence), 6),
                        "confirmation_time_seconds_from_event_start": round(
                            float((confirmed_at or now) - (event_start or now)), 3
                        ),
                        "model_device": runtime.resolved_device,
                        "frame_count": len(recording_frames),
                        "classes": list(OPENVINO_LABELS),
                    }
                    path = write_event(output_dir, event_index, event_label, recording_frames, measured_fps, metadata)
                    print(f"SAVED {path}")
                    recording = False
                    event_label = None
                    recording_frames = []
                    event_start = None
                    confirmed_at = None
                    below_since = None

            if not args.no_preview:
                preview = draw_overlay(frame, observation, recording, measured_fps, event_label)
                cv2.imshow("OpenVINO Action Capture", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if recording and recording_frames and event_label:
            event_index += 1
            metadata = {
                "label": event_label,
                "duration_seconds": round(recording_frames[-1].timestamp - recording_frames[0].timestamp, 3),
                "pre_buffer_seconds": args.pre_buffer,
                "closed_by_user": True,
                "model_device": runtime.resolved_device,
                "frame_count": len(recording_frames),
                "classes": list(OPENVINO_LABELS),
            }
            print(f"SAVED {write_event(output_dir, event_index, event_label, recording_frames, measured_fps, metadata)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
