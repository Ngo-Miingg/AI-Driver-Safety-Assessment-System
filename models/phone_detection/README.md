# Phone Detection Models

This folder documents the trained phone detector models used by the project.
The `.pt` files currently remain in `models/` for backward-compatible config
paths, while the application exposes them through a shared model registry.

## Realtime / Phone Test Options

| UI option | Backend key | Model file | Role |
|---|---|---|---|
| RT-DETR-L serious fine-tuned | `rtdetr_finetuned` | `models/phone_rtdetr_l.pt` | Main selected phone detector. Highest current mAP50-95. |
| RT-DETR-L comparison checkpoint | `rtdetr_light_compare` | `models/phone_rtdetr_l_light_compare.pt` | Comparison checkpoint from the light RT-DETR experiment. |
| YOLO26n fine-tuned | `yolo_finetuned` | `models/phone_yolo26n.pt` | Lightweight trained baseline and realtime fallback. |

## Compatibility File

| File | Note |
|---|---|
| `models/phone_yolo11n.pt` | Compatibility copy for earlier YOLO config/workflow names. It is not shown as a primary realtime choice. |

## Current Behavior Model

Realtime behavior recognition currently uses the built-in MediaPipe/OpenCV
pipeline:

- MediaPipe Face Mesh landmarks
- EAR threshold for eye closure
- MAR threshold for yawn candidates
- heuristic head pose / looking-away / head-down logic
- temporal thresholds from Settings

A trainable action-recognition behavior model is a future extension, not part
of the current trained model set.
