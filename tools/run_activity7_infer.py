from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from mddad.activity7_infer import Activity7Package
from mddad.multimodal_analysis import _clip_windows


def _video_duration_seconds(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()
    return frames / fps if fps > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Run Activity7 multimodal inference on a video file.")
    parser.add_argument("video", type=Path, help="Path to an input video file.")
    parser.add_argument(
        "--package",
        type=Path,
        default=Path(Config.ACTIVITY7_MODEL_PACKAGE_PATH),
        help="Path to activity7_final_best_model_package.zip or extracted folder.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, usually cpu or cuda.")
    parser.add_argument("--clip-seconds", type=float, default=Config.MULTIMODAL_CLIP_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=Config.MULTIMODAL_CLIP_STRIDE_SECONDS)
    parser.add_argument("--frame-count", type=int, default=Config.MULTIMODAL_FRAME_COUNT)
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    if not args.package.exists():
        raise SystemExit(f"Model package not found: {args.package}")

    package = Activity7Package(args.package, device=args.device)
    try:
        duration_seconds = _video_duration_seconds(args.video)
        windows = _clip_windows(duration_seconds, args.clip_seconds, args.stride_seconds)
        predictions = []
        for start_seconds, end_seconds in windows:
            clip = package.predict_clip(
                args.video,
                sample_frames=args.frame_count,
                start_seconds=start_seconds,
                duration_seconds=max(end_seconds - start_seconds, 0.5),
            )
            pred = clip.prediction
            predictions.append(
                {
                    "start_seconds": clip.start_seconds,
                    "end_seconds": clip.end_seconds,
                    "label": pred.label,
                    "confidence": pred.confidence,
                    "audio_track_detected": clip.audio_track_detected,
                    "sampled_frame_count": clip.sampled_frame_count,
                    "pose_frame_count": clip.pose_frame_count,
                    "top3": pred.probabilities[:3],
                }
            )

        top = max(predictions, key=lambda item: item["confidence"]) if predictions else None
        payload = {
            "video": str(args.video),
            "package": str(args.package),
            "duration_seconds": duration_seconds,
            "clip_seconds": args.clip_seconds,
            "stride_seconds": args.stride_seconds,
            "frame_count": args.frame_count,
            "top_prediction": top,
            "predictions": predictions,
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return

        print(f"Video: {args.video}")
        print(f"Package: {args.package}")
        print(f"Duration: {duration_seconds:.2f}s")
        if top:
            print(f"Top prediction: {top['label']} ({top['confidence']:.4f})")
        print("Clip predictions:")
        for item in predictions:
            print(
                f"  - {item['start_seconds']:.1f}s -> {item['end_seconds']:.1f}s: "
                f"{item['label']} ({item['confidence']:.4f})"
            )
    finally:
        package.close()


if __name__ == "__main__":
    main()
