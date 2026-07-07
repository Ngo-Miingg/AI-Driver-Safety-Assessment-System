# Demo Mode

## Recommended Settings

- Realtime target FPS: `8`
- Snapshot FPS: `2` or `3`
- Snapshot width: `360`
- Realtime YOLO: `0` unless testing phone detection specifically
- Uploaded sample FPS: `5`

## Demo Preparation

1. Prepare one normal driving video.
2. Prepare one drowsiness/eye closure video.
3. Prepare one yawning video.
4. Prepare one looking away/head down video.
5. Prepare one phone usage video if a detector model is available.

## Camera Verification

1. Log in.
2. Open `/debug/camera`.
3. Confirm `opened=true` and `frame_read=true`.
4. Close other apps that may use the webcam.

## Database Reset

For a clean demo, either use a fresh SQLite database path or manually clear old sessions through the UI/workflow. Do not delete database files while the app is running.

## Sample Workflow

Login -> select driver -> Dashboard -> Realtime Monitoring -> Stop -> Session Detail -> Upload Video -> Full AI Analysis -> History -> Evaluation -> CSV Export -> Settings.
