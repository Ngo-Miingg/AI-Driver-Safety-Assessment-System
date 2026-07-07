# Offline Deployment

## Offline Assets

The app avoids hard CDN dependency for realtime monitoring:

- Socket.IO client is served from `static/vendor/socket.io.min.js`.
- CSS and JavaScript are local under `static/`.

## Offline Run Steps

1. Install Python dependencies ahead of time.
2. Copy optional model files into `models/`.
3. Copy `.env.example` to `.env` and adjust paths.
4. Run:

```powershell
python app.py
```

5. Open:

```text
http://127.0.0.1:5000
```

## Optional Model Files

The app works without trained phone models. Advanced phone detection can be enabled later by placing one of:

```text
models/yolo11n.pt
models/phone_yolo11n.pt
models/phone_rtdetr_r18.pt
```

Then set:

```text
PHONE_DETECTOR_BACKEND=yolo_finetuned
```

or:

```text
PHONE_DETECTOR_BACKEND=rtdetr_finetuned
```

## Known Offline Limits

- Python packages must already be installed.
- Camera permissions still depend on the operating system.
- Model files are not downloaded automatically.
