# Project Cleanup Audit

Audit date: 2026-08-05

## Product boundary

The runtime is no longer a generic sandbox for every experiment. It now needs a
clear product boundary.

Current default product path:

```text
Camera -> Face detector -> Face landmarks/features -> Drowsiness model
       -> Optional phone detector -> Optional action model
       -> Temporal logic -> Session/report persistence
```

Active product-facing components:

| Capability | Status | Evidence |
|---|---|---|
| Local accounts and drivers | Working | Flask auth + SQLite |
| Realtime camera preview | Working | `realtime_routes.py` MJPEG feed |
| Face detection | Active | `models/driver_safety_v1/face_detection.pt` |
| Face feature extraction | Active | MediaPipe Face Mesh |
| Temporal drowsiness inference | Active | scaler + `best_model.h5` |
| Phone warning detector | Active | `models/phone_rtdetr_l.pt` |
| Action context model | Active but needs field validation | OpenVINO ADAS 0002 |
| Session/event/report persistence | Active | sessions, events, reports tables |

Research and lab components still present in source tree:

| Component | Current state | Recommended cleanup state |
|---|---|---|
| MMRT full v2 bundle | Source/artifacts still present | move to legacy research zone |
| Activity7 package | Source/artifacts still present | move to legacy research zone |
| VITDD behavior branch | Source references still present | move to legacy research zone |
| old multimodal test pages | partially gated | keep hidden or archive |
| old analysis routes | disabled by flash-return | archive or hard-disable from registration |

## Main problems found

1. The repository has a product/runtime story, but the codebase still carries a
   research-lab structure.
2. `config.py` still exposes many legacy knobs:
   `MMRT_*`, `ACTIVITY7_*`, `VITDD_*`, `YOLO_*`, old multimodal labels.
3. `analysis_routes.py` is still registered only through legacy gating, but it
   contains many dead branches that return early with "disabled" messages.
4. Main docs were inconsistent with the real runtime state. `README.md` was
   also broken by encoding issues.
5. Models on disk and model choices in code are not yet reduced to a clean
   handoff subset.

## Cleanup target

The clean handoff version should contain only these runtime model families:

- `driver_safety_v1`
- `openvino_driver_action_adas_0002`
- `phone_rtdetr_l.pt`

Everything else should become one of:

- archived research artifact
- disabled legacy route
- removed from product UI
- removed from default config

## What should stay

Keep in active runtime:

- `app.py`
- `config.py`
- `mddad/driver_safety_v1.py`
- `mddad/driver_action_openvino.py`
- `mddad/face_identity.py`
- `mddad/realtime_monitor.py`
- `mddad/realtime_routes.py`
- `mddad/sockets.py`
- `templates/driver_safety_realtime.html`
- `static/js/realtime.js`
- core dashboard / drivers / history / reports / evaluation flow

Keep, but outside default runtime story:

- report source
- Kaggle notebooks
- old experiment scripts
- legacy multimodal demo pages

## What should be cleaned next

### Safe cleanup

- remove legacy text from navigation, labels and help copy
- reduce config surface to product settings first, legacy settings second
- rewrite model inventory so runtime models are obvious
- separate runtime docs from research docs

### Structural cleanup

- move MMRT / Activity7 / VITDD routes behind one clear `legacy_research`
  package or stop importing them entirely
- move one-off builder scripts out of the top-level root into a dedicated
  archive/tools area
- split runtime model verification from report-generation tooling

### Handoff cleanup

- exclude `.venv_broken/`, `tmp/`, generated images, slide exports and local
  media from the handoff package
- ship runtime source and runtime models as two separate bundles
- include one operator guide with exact run/test steps

## Repository weight

Large directories that should not be treated as core source:

```text
dataset/          raw or prepared training data
uploads/          runtime user data
instance/         local SQLite database
models/           runtime + research weights
docs/overleaf_*   report package
tmp/              temporary local outputs
.venv_broken/     stale environment
```

## Immediate decisions

- `README.md` has been rewritten to reflect the real product runtime and fix the
  broken text encoding.
- Cleanup from this point should prefer isolation over deletion: first hide,
  gate or archive; then remove only after the runtime path is stable.

## Known technical debt

1. Camera preview ownership is still split between idle MJPEG preview and active
   inference capture.
2. Realtime metrics and model outputs need more hardware soak testing.
3. Action-model labels and phone warnings still need practical validation in
   cabin conditions, not just load-time verification.
4. Face identity enrollment exists as a capability path, but still needs full
   operational hardening before it can be treated as finished.
5. The source tree still contains too many top-level generated assets and
   one-off scripts for a clean third-party handoff.
