# AI Pipeline

## MediaPipe Signals

MediaPipe Face Mesh produces face landmarks. The app derives:

- EAR for eye closure.
- MAR for yawn candidates.
- Head pose heuristic for looking away/head down.
- Face detected / face missing state.

These signals are rule-based and do not require model training.

## Phone Detector

Phone detection is optional. The detector plug-in layer supports:

- `mediapipe_only`
- `yolo_pretrained`
- `yolo_finetuned`
- `rtdetr_finetuned`

Raw phone detections are filtered through:

- phone class check,
- configurable driver ROI,
- temporal confirmation,
- severity scaling by duration, count, and confidence.

Face-phone proximity is a planned enhancement. Current detector outputs do not include a reliable face bounding box in the phone pipeline.

## Event Tracking

Frame signals become temporal events only after minimum duration or repeated detection requirements are met. Events then contribute to risk score, safety score, alert level, fatigue trend, and report text.
