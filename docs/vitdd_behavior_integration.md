# ViT-DD Behavior Recognition Integration

## Purpose

This adds the second branch of the multimodal system:

- Phone branch: object detection, outputs phone bounding boxes and confidence.
- Behavior branch: ViT-DD-style driver behavior recognition, outputs behavior class probabilities.

## App integration

- Route: `/behavior-test`
- Model path: `models/vitdd_behavior.pt`
- Config keys:
  - `VITDD_MODEL_PATH`
  - `VITDD_MODEL_NAME`
  - `VITDD_IMAGE_SIZE`
  - `VITDD_SAMPLE_FPS`
  - `VITDD_LABELS`

## Model

The local adapter uses a Vision Transformer classifier through `timm`, matching the practical ViT-DD integration path. The original ViT-DD paper/source is used as the reference method, while this adapter keeps the Flask app stable and easy to test.

## Not yet connected

The behavior branch is not yet fused into realtime monitoring or uploaded-video Full AI reports. It is isolated first so the model can be trained, validated, and demonstrated safely.
