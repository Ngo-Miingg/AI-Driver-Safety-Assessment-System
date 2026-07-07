# Manual Test Checklist

## Setup

- [ ] Install dependencies with `pip install -r requirements.txt`.
- [ ] Copy `.env.example` to `.env` if custom paths are needed.
- [ ] Run `python app.py`.
- [ ] Open `http://127.0.0.1:5000/health`.

## Account And Drivers

- [ ] Log in with `admin` / `admin123`.
- [ ] Create a driver.
- [ ] Select an active driver.
- [ ] View driver detail.

## Upload

- [ ] Upload a valid MP4.
- [ ] Reject an invalid extension.
- [ ] Run metadata analysis.
- [ ] Run simulated analysis.
- [ ] Run MediaPipe analysis.
- [ ] Run Full AI with missing model and confirm warning.
- [ ] Run Full AI with model if available.

## Realtime

- [ ] Open `/debug/camera`.
- [ ] Open `/realtime`.
- [ ] Start monitoring.
- [ ] Confirm status is Running.
- [ ] Confirm elapsed time increments.
- [ ] Confirm behavior indicators update.
- [ ] Confirm snapshot preview updates if enabled.
- [ ] Stop monitoring.
- [ ] Open saved session detail.

## Reporting

- [ ] Check Dashboard metrics.
- [ ] Check History rows.
- [ ] Check Session Detail defense page.
- [ ] Check Reports page.
- [ ] Check Evaluation page.
- [ ] Export CSV.

## Settings

- [ ] Save threshold changes.
- [ ] Reset defaults.
- [ ] Confirm realtime settings affect new sessions.
