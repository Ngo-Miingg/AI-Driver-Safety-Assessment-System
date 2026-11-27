# Handoff for a new developer

## Start here

The product runtime is `driver_safety_v1`. Read, in order:

1. `README.md`
2. `models/driver_safety_v1/README.md`
3. `docs/PROJECT_CLEANUP_AUDIT.md`
4. `docs/PRODUCT_ROADMAP.md`

Do not infer current functionality from old MMRT, Activity7, phone-detector or
report files. They are retained as research history and are disabled by default.

## Local setup

```powershell
cd D:\Work\HocKy3\ComputerVision\MDDAD
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools\verify_driver_safety_v1.py --load-detector
python -m unittest discover -s tests -v
python app.py
```

Open `http://127.0.0.1:5000`. The local demo account is
`admin / admin123` unless `.env` overrides it.

## Model bundle

Required for realtime drowsiness:

```text
models/driver_safety_v1/face_detection.pt
models/driver_safety_v1/best_model.h5
models/driver_safety_v1/scaler.joblib
models/driver_safety_v1/manifest.json
```

Preserved but not integrated:

```text
models/driver_safety_v1/face_verification.h5
```

The verifier must not be presented as working identity verification until
enrollment, preprocessing, distance threshold and genuine/impostor evaluation
are implemented.

## Supported demo

1. Log in.
2. Select an active driver.
3. Open `/debug/camera` and confirm the configured camera reads a frame.
4. Open `/realtime` and start monitoring.
5. Confirm face detection, six-feature extraction, the `15 / 15` buffer and a
   finite drowsiness probability.
6. Stop and open the saved session.

Uploaded-video storage and metadata work, but uploaded-video inference has not
yet been ported to `driver_safety_v1`. Do not use old multimodal analysis as
evidence for the current product.

## Data and source package

Do not include these in a source handoff:

```text
.venv/
dataset/
instance/
logs/
uploads/
tools/car_logo_dataset/
```

Ship model weights separately from source. Never delete local SQLite data,
uploads or model artifacts through a generic cleanup command.

## Final checks

```powershell
python -m compileall -q mddad config.py app.py tools\verify_driver_safety_v1.py
python -m unittest discover -s tests -v
python tools\cleanup_handoff.py
```

Hardware release checks still required: repeated camera start/stop, missing
camera, missing face, low light, glasses and a long normal-driving soak test.
