# Driver Safety v1 Artifacts

| Artifact | SHA-256 | Role |
|---|---|---|
| `face_detection.pt` | `F4619B3C2DD00A3C631197F77BB2C5AA2DD12E91856754EC1F03CC6FFB4DA16D` | YOLO face detector |
| `face_verification.h5` | `10F882A01C55D71D36EA2F8F25975413A3EFE0869BEFD6DDEB61C0F60F9887A0` | 128-D face embedding model |
| `best_model.h5` | `60A55D559F61F73B53EE08E54435ADA32C779615C29C48A61C2B40837D5F536B` | 15x6 temporal drowsiness model |
| `scaler.joblib` | `DBC2065FD8A893019C8B870C9F2C72C6F21A7AEDE74122A82B3B2034B159BBB4` | StandardScaler fitted on the six temporal input features |

The scaler contains six features and reports 5,860,680 fitted frame rows, equal
to 390,712 training sequences of 15 frames. Together with the 97,678 validation
sequences shown in the report confusion matrix, this reproduces the documented
80/20 split of 488,390 sequences.

The exact machine-readable contract is stored in `manifest.json`. Verify the
bundle before a demo or handoff:

```powershell
python tools/verify_driver_safety_v1.py --load-detector
```

The drowsiness artifact and scaler also match the original Git repository
commit `767c4eb` byte-for-byte. Runtime feature extraction follows that source,
including its eye, mouth, and head-pose landmark definitions.

`face_verification.h5` is integrated as an optional identity layer. A driver
must first enroll one clear face image from the driver detail page. The runtime
uses YOLO to crop the face, the H5 verifier to produce a 128-D L2-normalized
embedding, and Euclidean distance `<= 0.94` to mark the active driver as
verified. The verifier requires `tensorflow` and `tf-keras` because the released
H5 contains legacy Keras `TFOpLambda` layers.

Identity verification is deliberately separate from temporal drowsiness
classification. A verified face does not mean the driver is alert, and a
drowsiness event does not prove identity unless the verifier also reports a
match.
