# Driver Safety v1 runtime settings

| Setting | Default | Basis |
| --- | ---: | --- |
| `REALTIME_TARGET_FPS` | 30 | Report maps 15 frames to about 0.5 seconds at 30 FPS |
| `DRIVER_SAFETY_SEQUENCE_LENGTH` | 15 | Training contract |
| `DRIVER_SAFETY_INFERENCE_STRIDE` | 2 | Training sliding-window step |
| `DRIVER_SAFETY_DECISION_THRESHOLD` | 0.50 | Report evaluation rule |
| `DRIVER_SAFETY_ALERT_THRESHOLD` | 0.70 | Original `test_cam.py` warning rule |
| `DRIVER_SAFETY_EVENT_CONFIRM_SECONDS` | 3.0 | Original `test_cam.py` alarm delay |
| `DRIVER_SAFETY_FACE_CONFIDENCE` | 0.251 | Best F1 threshold reported for YOLOv8s |
| `DRIVER_SAFETY_FACE_IMGSZ` | 640 | Reported exported deployment input size |
| `DRIVER_SAFETY_FACE_DETECTOR_FPS` | 2.0 | Application optimization; detector is not a BiLSTM input |
| `DRIVER_SAFETY_FACE_VERIFY_THRESHOLD` | 0.94 | Euclidean threshold reported for the LFW pair evaluation |
| `DRIVER_ACTION_WINDOW_SIZE` | 16 | Open Model Zoo action model clip contract |
| `DRIVER_ACTION_LABEL_SMOOTHING_FRAMES` | 15 | Require a stable raw top class before accepting it |
| `DRIVER_ACTION_CONFIDENCE_THRESHOLD` | 0.45 | Application acceptance threshold; raw probabilities remain visible for diagnosis |
| `DRIVER_ACTION_EVENT_CONFIRM_SECONDS` | 2.0 | Continuous accepted action required before an event is logged |

The UI reports the observed feature FPS and temporal-window duration. On the
training assumption, a full window should be near 0.5 seconds. A materially
longer value indicates that the machine is not extracting landmarks quickly
enough; lowering preview load or moving to the TFLite/OpenVINO edge pipeline is
preferable to fabricating duplicate frames.

The model output is binary only:

- `0`: alert/normal, including talking, singing, and laughing in the source labels.
- `1`: drowsy, including drowsiness, sustained eye closure, yawning, and head nodding.

It cannot independently identify which of those cues caused a positive result.

## Event authority

The product pipeline has one event manager and deliberately separates early
evidence from confirmed behavior:

| Signal | Runtime role | Can create a confirmed event? |
| --- | --- | --- |
| `best_model.h5` | Binary temporal drowsiness probability | Yes, after the 3-second confirmation rule |
| `driver-action-recognition-adas-0002` | Nine-class clip action recognition | Yes, after class stability, confidence, and the 2-second confirmation rule |
| RT-DETR phone detector | Immediate phone-presence bbox and early warning | No |
| Face detector / verifier | Face localization and driver identity | No |

RT-DETR can raise the live risk display while a phone is visible, but only the
action model can confirm and store a phone-use or phone-call behavior event.
This prevents duplicate phone events and keeps the database semantics stable.

The camera preview contains bounding boxes only. Human-readable states and raw
diagnostic probabilities are rendered by the localized web UI, avoiding
OpenCV's lack of Vietnamese text support and keeping model output separate from
presentation.

## Face identity verification

The product runtime now supports driver identity enrollment and realtime
matching as a separate layer from drowsiness detection:

```text
driver profile image
  -> YOLO face crop
  -> face_verification.h5
  -> 128-D L2-normalized enrolled embedding

realtime camera frame
  -> YOLO face crop
  -> face_verification.h5
  -> 128-D live embedding
  -> Euclidean distance <= 0.94 means verified
```

The H5 verifier was saved with legacy Keras `TFOpLambda` layers. Use
`tensorflow` plus `tf-keras` in the runtime environment; plain Keras 3 cannot
load this artifact reliably. Enrollment is required per driver before realtime
can report `verified` or `mismatch`.
