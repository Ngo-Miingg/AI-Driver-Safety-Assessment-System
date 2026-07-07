# Multimodal Driver Behavior Model Plan

## Ket luan kien truc

He thong muc tieu phai la **multimodal model**, khong phai tap hop rule/threshold thu cong.

Model dau vao chinh la **video clip**, khong phai anh don le. Moi du doan hanh vi phai dua tren chuoi thoi gian.

Input cua model:

- RGB video frames: nhan dien ngu canh cabin, dau, mat, tay, vat the, dien thoai.
- Audio track: nhan dien co noi chuyen, hoi thoai, cuoc goi, tieng noi trong cabin.
- Skeleton/landmark sequence: khung xuong co the, mat, tay theo thoi gian de hoc tu the va dong tac.

Output cua model:

- normal/safe driving
- drowsy/eye closure
- yawning
- looking away
- head down
- phone usage
- phone call / talking on phone
- talking to passenger / conversation
- mixed-risk

Phone usage khong nen chi la object detection rieng le. Trong kien truc moi, phone la mot **hanh vi** duoc model hoc tu nhieu tin hieu: hinh anh dien thoai, tay dua len tai, huong dau/mat, va audio speech.

## Nguyen tac bat buoc

1. Khong dung image-only model lam model chinh.
2. Khong dung nguong EAR/MAR/head-pose thu cong lam quyet dinh chinh.
3. Khong bat nguoi dung chinh threshold de nhan dien hanh vi.
4. Phone detection RT-DETR/YOLO chi duoc xem la baseline, teacher label, hoac auxiliary feature neu can.
5. Model cuoi cung phai nhan clip va tra ve class/probability theo hanh vi.

## Kien truc de xuat

```text
Video file / realtime clip window
        |
        +--> RGB branch
        |       VideoMAE / SlowFast / TimeSformer-style video backbone
        |
        +--> Pose-skeleton branch
        |       Face + body + hand landmarks per frame
        |       ST-GCN / temporal transformer over keypoint sequence
        |
        +--> Audio branch
        |       log-mel / wav2vec-style / VAD-speech embedding
        |       speech/conversation/call signal over same time window
        |
        +--> Fusion module
                concat/cross-attention/late fusion
                |
                Multiclass or multilabel behavior classifier
```

Recommended practical path:

- RGB branch: start with VideoMAE or SlowFast because they are standard video-action-recognition backbones.
- Skeleton branch: extract landmarks first, then train ST-GCN/temporal transformer on landmark sequences.
- Audio branch: start with voice activity/speech embeddings; later improve to speaker/conversation classification.
- Fusion: start with late fusion MLP, then upgrade to transformer fusion after data is stable.

## Dataset yeu cau

Final dataset phai la video clip, moi sample gom:

```text
sample_id/
  clip.mp4
  audio.wav              optional neu audio da nam trong mp4 thi extract khi train
  landmarks.npy          generated cache, khong phai label thu cong
  label.json
```

Vi du `label.json`:

```json
{
  "labels": ["phone_call"],
  "start_sec": 12.0,
  "end_sec": 18.0,
  "driver_id": "optional",
  "notes": "driver holds phone near ear and speaks"
}
```

Label nen ho tro multilabel vi hanh vi co the chong len nhau:

- `phone_usage` + `looking_away`
- `phone_call` + `talking`
- `drowsy` + `head_down`

Dataset hien co `Roboflow Mobile-phone` la image detection dataset, khong du de train multimodal behavior model. No chi nen dung de:

- pretrain/teacher cho phone appearance
- tao pseudo-label phone tren video
- lam baseline so sanh

## Audio input

Audio khong can transcript noi dung rieng tu nguoi lai. Muc tieu la tin hieu hanh vi:

- co tieng noi hay khong
- speech duration trong clip
- co hoi thoai/cuoc goi hay khong
- speech energy/noise context

Trong realtime, audio co the di theo cua so 2-8 giay dong bo voi video. Trong upload analysis, extract audio tu video bang ffmpeg/moviepy roi dua vao audio branch.

## Skeleton input

Skeleton/landmark khong duoc dung nhu rule threshold cuoi cung. Chung la feature sequence cho model hoc.

Landmark nen gom:

- face landmarks: mat, mieng, dau
- body pose: vai, co, than tren
- hand landmarks: tay cam dien thoai, tay dua len tai, tay roi vo lang

Neu MediaPipe duoc dung, vai tro cua no la **feature extractor**, khong phai rule engine.

## Tich hop vao app hien tai

Nen giu cac luong cu de demo va fallback, nhung them mot workflow moi:

```text
Upload video
-> extract RGB clip windows
-> extract audio windows
-> extract skeleton windows
-> multimodal model inference
-> behavior events
-> risk/report/evaluation
```

Realtime:

```text
Webcam + microphone
-> rolling clip buffer
-> extract RGB/audio/skeleton
-> multimodal inference per window
-> realtime_update/realtime_event
```

UI wording:

- Model chinh: `Multimodal video-audio-skeleton behavior model`
- Baseline cu: `Legacy MediaPipe/OpenCV rules`
- Phone detector rieng: `Auxiliary phone detector / baseline`, khong goi la model chinh.

## Training plan

Phase 1: Video-only behavior baseline

- Dung clip video, khong dung anh don.
- Train VideoMAE/SlowFast voi label hanh vi.
- Phone usage la mot class hanh vi, khong chi bounding box.

Phase 2: Skeleton branch

- Extract face/body/hand landmarks thanh `.npy`.
- Train ST-GCN/temporal transformer.
- So sanh video-only vs skeleton-only.

Phase 3: Audio branch

- Extract audio tu video.
- Train speech/call/conversation signal.
- Khong can transcript noi dung neu de tai chi can nhan dien noi chuyen/goi dien.

Phase 4: Fusion

- Fusion RGB + skeleton + audio.
- Bao cao ablation:
  - RGB only
  - skeleton only
  - audio only
  - RGB + skeleton
  - RGB + audio
  - RGB + skeleton + audio

Phase 5: App integration

- Them `analysis_mode = multimodal`.
- Luu model backend vao session metadata.
- Full AI moi goi multimodal inference.
- Realtime dung rolling window, khong phan tich tung frame don le.

## Diem can noi trong bao cao

He thong ban dau dung MediaPipe/OpenCV rules va detector dien thoai de tao baseline thuc nghiem. Tuy nhien, phien ban muc tieu duoc thiet ke lai thanh multimodal behavior recognition system. Model nhan video clip, audio va skeleton sequence, sau do tu hoc bieu dien hanh vi nguy hiem ma khong phu thuoc vao nguong thu cong.

## Tai lieu tham khao

- Hugging Face VideoMAE documentation: https://huggingface.co/docs/transformers/en/model_doc/videomae
- Hugging Face video classification guide: https://huggingface.co/docs/transformers/en/tasks/video_classification
- PyTorchVideo SlowFast documentation: https://pytorchvideo.readthedocs.io/en/latest/api/models/slowfast.html
- Google MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- ST-GCN paper: https://arxiv.org/abs/1801.07455
- Silero VAD repository: https://github.com/snakers4/silero-vad
- pyannote.audio repository: https://github.com/pyannote/pyannote-audio
- Whisper repository: https://github.com/openai/whisper
