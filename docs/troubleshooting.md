# Troubleshooting

## Login Fails

- Confirm `SEED_DEMO_USER=1`.
- Use `admin` / `admin123` unless `.env` changed the demo credentials.

## Realtime Does Not Start

- Open browser DevTools and check SocketIO logs.
- Open `/debug/camera`.
- Close other apps using the webcam.
- Confirm `static/vendor/socket.io.min.js` loads.

## Snapshot Preview Is Slow

- Set `Snapshot FPS` to `2` or `3`.
- Set `Snapshot width` to `360`.
- Keep realtime YOLO disabled.

## Full AI Shows YOLO Warning

- This is expected if the model file is missing.
- Place the model under `models/`.
- Confirm `PHONE_DETECTOR_BACKEND` and model path match.

## RT-DETR Does Not Load

- Confirm Ultralytics is installed.
- Confirm the model exists at `models/phone_rtdetr_r18.pt`.
- Set `PHONE_DETECTOR_BACKEND=rtdetr_finetuned`.

## Upload Analysis Is Slow

- Lower uploaded sample FPS in Settings.
- Use short 15-30 second demo videos.
