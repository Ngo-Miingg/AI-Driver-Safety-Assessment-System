# Model Guide

## MediaPipe

MediaPipe does not require training for this project.

The system uses MediaPipe FaceMesh landmarks to estimate:

- EAR for eye closure.
- MAR for yawning.
- Simple head orientation heuristics.
- Face presence / missing face.

Drowsiness and distraction are rule-based because this is explainable, fast, and suitable for a local defense demo. The system converts landmark signals into temporal events using configurable thresholds.

## YOLO Pretrained Model

Default pretrained YOLO path:

```text
models\yolo11n.pt
```

Use this for:

- Uploaded-video Full AI phone detection.
- Optional realtime phone detection.
- Quick demo without training.
- General object detection where the phone class is already recognized.

If the model is missing, uploaded Full AI stores a warning and realtime monitoring continues without crashing.

## Optional Fine-Tuned Model

Recommended optional fine-tuned path:

```text
models\phone_yolo11n.pt
```

Fine-tuning is optional. It is useful only if pretrained YOLO fails to detect phones reliably in your demo camera angle, lighting, or local phone style.

## When To Use Pretrained YOLO

Use pretrained YOLO when:

- You need a fast defense-ready setup.
- Phone visibility is clear.
- The phone is not too small or hidden.
- General phone detection is good enough.

## When To Fine-Tune

Fine-tune only when:

- Phone detection is unreliable in your prepared videos.
- The camera angle is unusual.
- The phone is often partially hidden.
- You need more consistent phone detection metrics.

Training is not mandatory for this project.

## Recommended Dataset Structure

If fine-tuning is done outside the current app, use a YOLO dataset structure similar to:

```text
datasets\phone_detection
|-- images
|   |-- train
|   |-- val
|   `-- test
|-- labels
|   |-- train
|   |-- val
|   `-- test
`-- data.yaml
```

Example `data.yaml`:

```yaml
path: D:/Work/HocKy3/ComputerVision/MDDAD/datasets/phone_detection
train: images/train
val: images/val
test: images/test
names:
  0: phone
```

Label format should be standard YOLO bounding boxes:

```text
class_id center_x center_y width height
```

All coordinates are normalized from 0 to 1.

## Where To Place Trained Weights

After training, copy the best weights to:

```text
models\phone_yolo11n.pt
```

Keep the pretrained model too:

```text
models\yolo11n.pt
```

## How To Switch Model Path

For uploaded Full AI:

```powershell
$env:YOLO_MODEL_PATH="D:\Work\HocKy3\ComputerVision\MDDAD\models\phone_yolo11n.pt"
```

For realtime YOLO:

```powershell
$env:REALTIME_YOLO_MODEL_PATH="D:\Work\HocKy3\ComputerVision\MDDAD\models\phone_yolo11n.pt"
```

You can also use the Settings page for realtime YOLO enable/FPS/confidence. The current model path is displayed there for reference.

## Why Drowsiness Is Rule/Landmark-Based

The project uses MediaPipe landmarks plus rules instead of a trained drowsiness model because:

- No custom dataset is required.
- Behavior is easier to explain in a defense.
- Thresholds are visible and configurable.
- It runs locally on ordinary hardware.
- It avoids overclaiming machine-learning accuracy.

Future work may add a trained classifier, but the current system is intentionally transparent: landmarks become signals, signals become events, events become scores and reports.
