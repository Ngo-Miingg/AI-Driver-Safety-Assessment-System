# Deployable Model Inventory

## Retained

| Artifact | Runtime role |
|---|---|
| `driver_safety_v1/face_detection.pt` | Face localization for Driver Safety v1 |
| `driver_safety_v1/best_model.h5` | Temporal binary drowsiness classification |
| `driver_safety_v1/scaler.joblib` | Driver Safety v1 feature normalization |
| `driver_safety_v1/face_verification.h5` | Driver face enrollment and realtime identity verification |
| `phone_rtdetr_l.pt` | Early phone-presence warning and bounding-box evidence |
| `openvino_driver_action_adas_0002/FP16-INT8/*` | Clip-level driver action recognition |

## Removed from deployment

- MMRT multimodal full v2 t4x2
- Activity7 multimodal
- RT-DETR-L comparison checkpoint

Research reports and source code may still mention removed models to preserve
the history and reproducibility of earlier experiments. They must not be
presented as available runtime backends.
