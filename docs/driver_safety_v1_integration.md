# Driver Safety v1 integration

This document records the model contract verified against:

- `Nhom01_BaoCaoDATN.pdf`, sections 4.4.1 to 4.4.3.
- The original training repository `cUddddD/Drowsiness_Detection` at commit
  `767c4eb` (the last commit that still contains the released artifacts).
- SHA-256 hashes in `models/driver_safety_v1/manifest.json`.

The local `best_model.h5` and `scaler.joblib` are byte-for-byte identical to
the files in that source commit.

The repository is useful evidence but is not fully reproducible without a
small repair: its committed `process_frame()` returns `(features, landmarks)`,
while `preprocess.py` appends that tuple as if it were only the numeric feature
vector. Rebuilding the dataset requires explicitly unpacking `features, _`.
The integration therefore follows the extractor and released inference demo,
but does not claim that the repository can retrain the model unchanged.

## Active inference path

```text
camera frame (target 30 FPS)
  +-> MediaPipe Face Mesh
  |     -> source-exact [EAR_Left, EAR_Right, MAR, Pitch, Yaw, Roll]
  |     -> original StandardScaler
  |     -> overlapping 15-frame window, inference stride 2
  |     -> best_model.h5 -> P(drowsy)
  |
  +-> face_detection.pt at a reduced display cadence
        -> face bounding box for localization and identity crop
        -> face_verification.h5 -> live 128-D embedding when the driver is enrolled

P > 0.50 -> model-level drowsiness decision
P > 0.70 continuously for 3 s -> application warning and logged event
```

YOLO is deliberately not used as a gate for the temporal classifier. The
training pipeline obtained its six features directly from Face Mesh. Running
the 64 MB YOLOv8s detector on every frame both reduces temporal sampling rate
and makes a detector miss erase otherwise valid landmarks. The runtime samples
YOLO at 2 FPS for localization while Face Mesh feeds every valid camera frame
to the BiLSTM.

## Exact six-feature contract

The formulas and landmark indices in `mddad/driver_safety_v1.py` mirror
`datapreprocessing/utils.py` from the source repository. Important details are:

1. Left EAR uses landmarks 362, 385, 387, 263, 373, and 374 through the
   repository's 16-point eye ordering.
2. Right EAR uses landmarks 133, 160, 158, 33, 144, and 145 through the same
   ordering.
3. MAR uses mouth width 61-291 and the average of vertical distances 0-17 and
   40-37. It is not the simpler 13-14 / 78-308 ratio.
4. Head pose uses points 4, 152, 263, 33, 61, and 291 with the exact 3-D face
   template from the source.
5. Feature order is fixed. The persisted scaler must transform the 15x6 array
   before it is reshaped to `(1, 15, 6)`.

If valid landmarks disappear for a meaningful gap, the temporal buffer is
cleared. This prevents frames from before and after a camera/face loss from
being compressed into one artificial sequence.

## Artifact roles

| Artifact | Verified role | Deployment status |
| --- | --- | --- |
| `face_detection.pt` | YOLOv8s, one class `face`; report threshold 0.251 | Active at low cadence, 640 input for the large in-cabin face |
| `best_model.h5` | Binary BiLSTM, input `(15, 6)`, sigmoid output | Active |
| `scaler.joblib` | Original six-feature StandardScaler | Active and mandatory |
| `face_verification.h5` | MobileNetV2 128-D L2-normalized face embedding | Active after driver face enrollment |

The report gives an identity matching threshold of Euclidean distance 0.94.
That number came from 2,000 generated LFW pairs and must not be presented as a
production identity guarantee. The current runtime supports guided multi-sample
enrollment, stores the averaged embedding centroid, and compares live embeddings
during realtime monitoring. A stronger handoff still requires a driver-disjoint
acceptance set and anti-spoofing/liveness checks before any production claim.

## Evidence and limitations

The drowsiness report evaluates a random stratified 80/20 split of overlapping
windows and reports TN=40,518, FP=1,998, FN=2,044, TP=53,118 and ROC AUC 0.9938.
Those results validate the released split, but overlapping windows and possible
subject overlap can make generalization appear stronger than a subject-disjoint
test. The Flask runtime must therefore remain a monitored warning aid, not a
certified safety controller.

The report also contains two face-detector mAP@50 values (approximately 0.78 in
one discussion and 0.699 in another). The runtime adopts only the unambiguous
deployment facts: one `face` class, confidence 0.251, and 640-pixel exported
input guidance. It does not repeat either mAP as a resolved benchmark.

Run the artifact and shape acceptance check before every handoff:

```powershell
python tools/verify_driver_safety_v1.py --load-detector
```
