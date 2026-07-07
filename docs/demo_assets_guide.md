# Demo Assets Guide

Prepare a small set of videos before the defense. Keep the files local and short enough that analysis finishes quickly.

## Recommended Folder

Store demo videos outside generated upload storage, for example:

```text
D:\Work\HocKy3\ComputerVision\MDDAD\demo_assets
```

The app will copy uploaded files into `uploads/` during testing.

## Naming Convention

Use clear names:

```text
01_normal_driver.mp4
02_eye_closure_drowsiness.mp4
03_yawning.mp4
04_looking_away_head_down.mp4
05_phone_usage.mp4
06_mixed_risk.mp4
```

## Recommended Video Length

- Minimum: 20 seconds.
- Preferred: 30-90 seconds.
- Avoid very long files during live defense because analysis is synchronous.

## Recommended Lighting

- Use bright, even lighting.
- Avoid strong backlight.
- Keep the face visible.
- Avoid heavy shadows across the eyes and mouth.

## Webcam Positioning

- Place the camera near dashboard/eye level.
- Keep the full face visible.
- Avoid extreme side angles unless testing looking-away behavior.
- Keep the phone visible if testing phone usage.

## File Formats

Recommended:

- `.mp4`

Accepted:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`

## Asset Types

### 1. Normal Driver Video

Purpose: baseline comparison.

Content:

- Face visible.
- Eyes open.
- Looking forward.
- No phone.
- Minimal risk behavior.

Expected result:

- Low event count.
- High safety score.
- Normal or low alert.

### 2. Eye Closure / Drowsiness Video

Purpose: demonstrate EAR-based drowsiness.

Content:

- Face visible.
- Eyes closed longer than the configured eye closure minimum duration.
- Repeat two or three times if possible.

Expected result:

- `eye_closure` events.
- Drowsiness contribution.
- Lower safety score than normal video.

### 3. Yawning Video

Purpose: demonstrate MAR-based yawn detection.

Content:

- Face visible.
- Mouth open widely for more than yawn minimum duration.
- Avoid covering the mouth.

Expected result:

- `yawn` events.
- Fatigue-related recommendations.

### 4. Looking Away / Head Down Video

Purpose: demonstrate distraction signals.

Content:

- Look left/right for several seconds.
- Look down for several seconds.
- Keep face mostly visible.

Expected result:

- `looking_away` and/or `head_down` events.
- Distraction contribution.

### 5. Phone Usage Video

Purpose: demonstrate YOLO phone detection.

Content:

- Hold a phone clearly in view.
- Keep it visible for several sampled frames.
- Avoid tiny or hidden phone positions.

Expected result:

- `phone_usage` events if YOLO model is available.
- Phone usage count and phone recommendations.

### 6. Mixed-Risk Video

Purpose: final demo and scoring explanation.

Content:

- Combine yawning, looking away, head down, and phone visibility.
- Keep duration around 45-90 seconds.

Expected result:

- Multiple event types.
- Higher risk score.
- Stronger report explanation.

## Practical Demo Advice

- Test every video before the defense.
- Keep at least one video that works without YOLO.
- Keep one phone usage video for YOLO demo.
- Keep a backup short video in case analysis takes too long.
- Do not rely only on live webcam behavior; prerecorded videos are more repeatable.
