# Báo cáo model và thư viện hiện tại

**Dự án:** AI Driver Safety Assessment System  
**Ngày rà soát:** 2026-08-07  
**Phạm vi:** code và artifacts thực tế trong `D:\Work\HocKy3\ComputerVision\MDDAD`

## 1. Kết luận

Runtime sản phẩm mặc định hiện tại là `driver_safety_v1`, không phải MMRT multimodal. Luồng đang chạy:

```text
Webcam -> YOLO face detector -> MediaPipe Face Mesh
       -> StandardScaler -> temporal drowsiness model
       -> OpenVINO driver action model -> temporal confirmation
       -> risk engine -> SQLite -> realtime UI/report/history
```

| Thành phần | Trạng thái |
|---|---|
| Face detector | Đang hoạt động |
| Face Mesh feature extractor | Đang hoạt động |
| Drowsiness model | Đang hoạt động |
| OpenVINO action model | Đang hoạt động |
| Face verification | Đang hoạt động khi tài xế đã enrollment |
| Event/risk/report/SQLite | Đang hoạt động |
| Audio vào Driver Safety v1 | Chưa nối |
| Body skeleton vào Driver Safety v1 | Chưa nối |
| Upload video product inference | Chưa nối, mới metadata |
| RT-DETR phone detector | Artifact còn, realtime mặc định tắt |
| MMRT/Activity7 | Nhánh legacy, route mặc định tắt |

## 2. Artifact hiện có

### 2.1. Driver Safety v1

Thư mục `models/driver_safety_v1/`:

| File | Vai trò | Contract |
|---|---|---|
| `face_detection.pt` | YOLO định vị khuôn mặt | frame BGR -> face box |
| `best_model.h5` | Classifier buồn ngủ theo thời gian | `(15,6)` -> xác suất |
| `scaler.joblib` | StandardScaler của training | 6 feature -> feature chuẩn hóa |
| `face_verification.h5` | Face embedding | `(112,112,3)` -> vector 128 chiều |
| `openvino/drowsiness.xml/.bin` | Bản OpenVINO của drowsiness | cùng contract với H5 |
| `openvino/face_verification.xml/.bin` | Bản OpenVINO verifier | face crop -> vector 128 chiều |
| `manifest.json` | Contract/provenance | metadata, không phải model |

Hash đã được ghi trong `models/driver_safety_v1/README.md`:

| File | SHA-256 |
|---|---|
| `face_detection.pt` | `F4619B3C2DD00A3C631197F77BB2C5AA2DD12E91856754EC1F03CC6FFB4DA16D` |
| `face_verification.h5` | `10F882A01C55D71D36EA2F8F25975413A3EFE0869BEFD6DDEB61C0F60F9887A0` |
| `best_model.h5` | `60A55D559F61F73B53EE08E54435ADA32C779615C29C48A61C2B40837D5F536B` |
| `scaler.joblib` | `DBC2065FD8A893019C8B870C9F2C72C6F21A7AEDE74122A82B3B2034B159BBB4` |

### 2.2. OpenVINO driver action model

Thư mục `models/openvino_driver_action_adas_0002/FP16-INT8/` chứa encoder và decoder:

```text
16 BGR frames
-> resize short side 224 + center crop 224x224
-> encoder input [1,3,224,224]
-> 16 embeddings x 512
-> decoder input [1,16,512]
-> logits [1,9] -> softmax
```

Class gốc và mapping product:

| ID | Open Model Zoo | Product class |
|---:|---|---|
| 0 | Safe driving | `normal_driving` |
| 1 | Texting left | `phone_texting` |
| 2 | Texting right | `phone_texting` |
| 3 | Talking phone left | `phone_calling` |
| 4 | Talking phone right | `phone_calling` |
| 5 | Operating radio | `operating_radio` |
| 6 | Drinking eating | `drinking_eating` |
| 7 | Reaching behind | `reaching_behind` |
| 8 | Hair and makeup | `hair_and_makeup` |

Các class left/right được gộp trước UI và event layer. Class public gồm:

```text
normal_driving, phone_texting, phone_calling,
operating_radio, drinking_eating, reaching_behind, hair_and_makeup
```

Mapping event:

| Product class | Event |
|---|---|
| `normal_driving` | Không tạo event |
| `phone_texting` | `phone_usage` |
| `phone_calling` | `phone_call` |
| `operating_radio` | `in_vehicle_display_interaction` |
| `drinking_eating` | `eating_distraction` |
| `reaching_behind` | `visual_posture_distraction` |
| `hair_and_makeup` | `visual_posture_distraction` |

### 2.3. RT-DETR phone detector

File `models/phone_rtdetr_l.pt`, backend Ultralytics RT-DETR, class `Mobile-phone`.

Vai trò được thiết kế là cảnh báo sớm và bounding box. Nó không được ghi đè class hành vi của OpenVINO action model. Hiện tại:

```text
PHONE_DETECTOR_BACKEND = disabled
REALTIME_PHONE_DETECTOR_BACKEND = disabled
```

Do đó phone texting/calling trong realtime chính phải đến từ action model, không đến RT-DETR.

### 2.4. MMRT/Activity7 legacy

Code còn `mmrt_v2_infer.py`, `activity7_infer.py`, `live_multimodal.py` và model-lab routes. Artifact hiện thấy:

```text
models/mmrt_artifacts_bundle/mmrt_multimodal_full_v1/best_full.pt
models/mmrt_artifacts_bundle/mmrt_multimodal_smoke/best_smoke.pt
```

Không thấy checkpoint `mmrt_multimodal_full_v2_t4x2` hoặc package Activity7 trong inventory hiện tại. `ENABLE_LEGACY_RESEARCH_ROUTES=0`, nên không phải runtime mặc định.

## 3. Pipeline Driver Safety v1

### 3.1. Face và drowsiness

`mddad/driver_safety_v1.py` thực hiện:

1. OpenCV đọc webcam frame.
2. Ultralytics YOLO tìm face box.
3. MediaPipe Face Mesh tạo landmark.
4. Trích sáu feature theo đúng thứ tự manifest:

```text
left_ear, right_ear, mar, pitch_degrees, yaw_degrees, roll_degrees
```

5. Chuẩn hóa bằng `scaler.joblib`.
6. Gom chuỗi 15 frame.
7. Drowsiness model trả xác suất.

Contract hiện tại:

| Tham số | Giá trị |
|---|---:|
| Sequence length | 15 frame |
| Feature count | 6 |
| Model-positive threshold | 0.5 |
| Alert threshold | 0.7 |
| Drowsiness confirmation | 3.0 s |

Nếu OpenVINO IR có đủ XML/BIN, drowsiness được compile bằng OpenVINO; nếu không, code fallback sang Keras H5.

### 3.2. Face behavior signals

`mddad/face_behavior_signals.py` tạo các tín hiệu bổ sung, không thay thế drowsiness model:

- `yawning`: MAR >= `0.60`.
- `looking_away`: yaw lệch khoảng `20°`, có release hysteresis khoảng `15°`.
- `head_down`: vertical offset >= `0.08`.

Các tín hiệu đi qua temporal confirmation:

```text
yawn 2.0 s, looking-away 2.0 s, head-down 2.0 s
```

### 3.3. Action recognition

`mddad/driver_action_openvino.py`:

- OpenVINO encoder/decoder.
- Window 16 frame.
- Sample stride 2 frame.
- Label smoothing 15 prediction.
- Confidence threshold chung 0.45.
- `drinking_eating` threshold riêng 0.70.
- Action event confirmation 2.0 s.

Phân biệt ba mức output:

```text
instant prediction -> smoothed/accepted class -> confirmed event
```

### 3.4. Event, risk, report

`DriverSafetyEventManager` theo dõi độc lập drowsiness, action, yawn, looking-away và head-down. Khi đủ thời gian xác nhận:

```text
confirmed event
-> contribution_for_event()
-> summarize_session()
-> events + risk_logs + report + session metrics
```

Minh chứng event mới:

```text
event finalized -> JPEG uploads/evidence/
-> events.evidence_image_path -> /evidence/<filename>
-> thumbnail trong Session Detail
```

## 4. Xác thực khuôn mặt

`mddad/face_identity.py` dùng verifier H5 hoặc OpenVINO IR:

```text
face enrollment
-> face crop 112x112 RGB
-> embedding 128-D
-> L2 normalize
-> lưu SQLite

realtime
-> YOLO face box -> embedding
-> Euclidean distance
-> distance <= 0.94 => verified
```

Identity verification độc lập với drowsiness: xác thực đúng tài xế không có nghĩa tài xế tỉnh táo.

## 5. Audio và skeleton

Code có SocketIO `realtime_audio_chunk`, nhưng chunk này chỉ được drain và đưa vào nhánh `multimodal_live` legacy. `_run_driver_safety_v1` hiện không đọc audio chunk để ra quyết định.

MediaPipe Pose và skeleton sequence cũng chỉ nằm trong loader MMRT/live multimodal legacy. Driver Safety v1 hiện dùng Face Mesh geometry, chưa dùng body skeleton.

Các thư viện audio có trong project/environment:

- `librosa`: load, resample, log-mel và audio features.
- `soundfile`: audio I/O.
- `sounddevice`: microphone support trong environment.
- `numpy`: audio buffer.

Kết luận: giao diện có thể có trạng thái mic/skeleton ở nhánh legacy, nhưng điều đó không chứng minh chúng đã đi vào model Driver Safety v1.

## 6. Thư viện

| Nhóm | Thư viện | Vai trò |
|---|---|---|
| Web | Flask | HTTP routes/templates |
| Realtime | Flask-SocketIO | Socket event/background session |
| Persistence | SQLite3 | Local database/offline storage |
| CV | OpenCV | Camera, frame, video, JPEG, stream |
| CV | MediaPipe | Face Mesh; Pose ở legacy |
| CV | Ultralytics | YOLO face detector, RT-DETR phone detector |
| DL | OpenVINO | Action, drowsiness IR, face verifier IR |
| DL | TensorFlow/tf-keras/Keras | H5 fallback và verifier |
| DL | PyTorch/torchvision | MMRT/Activity7 legacy |
| Audio | librosa/soundfile/sounddevice | Audio pipeline legacy/microphone |
| Data | NumPy | Tensor/features/probabilities |
| Data | scikit-learn/joblib | StandardScaler và serialization |
| Media | Pillow | Preprocessing/overlay |
| Web support | python-dotenv/Werkzeug/Jinja2 | config, upload, template |

Các phiên bản đã kiểm tra trong `.venv`:

```text
Flask 3.1.3
Flask-SocketIO 5.6.1
OpenCV 5.0.0.93
MediaPipe 0.10.14
Ultralytics 8.4.115
OpenVINO 2026.3.0
TensorFlow 2.18.1
tf-keras 2.18.0
Keras 3.15.1
PyTorch 2.11.0+cu128
Torchvision 0.26.0+cu128
NumPy 2.0.2
scikit-learn 1.6.1
librosa 0.11.0
```

## 7. Thiết bị suy luận

Kiểm tra runtime hiện tại:

```text
OpenVINO devices: CPU, GPU.0, GPU.1
Action model: ready on GPU.1
Drowsiness IR: ready on GPU.1
```

Các phần không đồng nhất hoàn toàn:

- OpenVINO action/drowsiness: GPU.1.
- YOLO face detector: `auto`, phụ thuộc PyTorch CUDA.
- MediaPipe Face Mesh: thường CPU/TFLite, không phải OpenVINO GPU.
- Face verifier: ưu tiên OpenVINO IR, fallback TensorFlow/Keras.
- MMRT legacy: config có thể CPU; route đang tắt.

## 8. Cấu hình quan trọng

| Key | Mặc định | Ý nghĩa |
|---|---:|---|
| `REALTIME_BEHAVIOR_MODEL_BACKEND` | `driver_safety_v1` | Backend realtime |
| `DRIVER_SAFETY_SEQUENCE_LENGTH` | 15 | Drowsiness window |
| `DRIVER_SAFETY_DECISION_THRESHOLD` | 0.5 | Model drowsiness |
| `DRIVER_SAFETY_ALERT_THRESHOLD` | 0.7 | Alert drowsiness |
| `DRIVER_SAFETY_EVENT_CONFIRM_SECONDS` | 3.0 | Xác nhận drowsiness |
| `DRIVER_ACTION_WINDOW_SIZE` | 16 | Action window |
| `DRIVER_ACTION_SAMPLE_STRIDE_FRAMES` | 2 | Lấy mẫu action |
| `DRIVER_ACTION_LABEL_SMOOTHING_FRAMES` | 15 | Làm ổn định class |
| `DRIVER_ACTION_CONFIDENCE_THRESHOLD` | 0.45 | Action threshold |
| `DRIVER_ACTION_DRINKING_EATING_THRESHOLD` | 0.70 | Threshold riêng |
| `DRIVER_ACTION_EVENT_CONFIRM_SECONDS` | 2.0 | Xác nhận action |
| `DRIVER_ACTION_DEVICE` | `auto` | OpenVINO device |
| `REALTIME_PHONE_DETECTOR_BACKEND` | `disabled` | RT-DETR realtime |
| `REALTIME_ENABLE_EVENT_EVIDENCE` | 1 | Lưu JPEG minh chứng |
| `ENABLE_LEGACY_RESEARCH_ROUTES` | 0 | Tắt MMRT/Activity7 routes |

## 9. Kiểm tra thực tế

Đã kiểm tra:

- `compileall mddad config.py app.py`: OK.
- `/dashboard`, `/drivers`, `/realtime`, `/history`, `/reports`, `/evaluation`, `/settings`, `/upload`: HTTP 200 sau login.
- `/realtime/feed`: trả `multipart/x-mixed-replace; boundary=frame`.
- Face detector, drowsiness model, scaler, face verifier: artifact available.
- OpenVINO encoder/decoder: ready.
- Database foreign-key check: không lỗi.
- Không có orphan events/reports.

## 10. Các điểm chưa hoàn thiện

1. Upload video chưa chạy product inference, mới lưu video và metadata.
2. Audio chưa nối vào Driver Safety v1.
3. Skeleton body chưa nối vào Driver Safety v1.
4. RT-DETR phone detector đang tắt trong realtime chính.
5. MMRT/Activity7 còn code nhưng không phải runtime mặc định.
6. Có session cũ còn trạng thái `active` sau các lần crash/dừng không hoàn tất; cần recovery khi làm sạch database.
7. Chưa có benchmark test-set mới chạy trực tiếp trên camera local; số liệu trong manifest là số liệu artifact/report.
8. MediaPipe là feature extractor, không phải action classifier.
9. Minh chứng chỉ có cho event được tạo sau khi bản lưu evidence chạy; event cũ không thể tạo lại ảnh.

## 11. Cách mô tả chính xác khi demo

Nên nói:

> Hệ thống triển khai pipeline Driver Safety v1 gồm định vị khuôn mặt bằng YOLO, trích xuất sáu đặc trưng hình học bằng MediaPipe Face Mesh, chuẩn hóa bằng StandardScaler, nhận diện buồn ngủ theo chuỗi thời gian và nhận diện hành vi tài xế bằng OpenVINO driver-action-recognition-adas-0002. Kết quả được ổn định bằng temporal confirmation trước khi ghi event, tính risk score và sinh báo cáo.

Không nên nói:

- Realtime hiện đã dùng audio và skeleton; thực tế v1 chưa nhận hai modality này.
- MMRT v2 đang chạy mặc định; checkpoint v2 không có trong inventory hiện tại.
- RT-DETR là model quyết định phone behavior; realtime mặc định đang tắt RT-DETR.
- MediaPipe tự nhận diện hành vi; MediaPipe chỉ tạo landmark/geometry signals.

## 12. File source trọng yếu

- `config.py`: đường dẫn model, device và threshold.
- `mddad/driver_safety_v1.py`: face, Face Mesh, scaler, drowsiness.
- `mddad/driver_action_openvino.py`: action encoder/decoder và class mapping.
- `mddad/face_identity.py`: face embedding/verification.
- `mddad/face_behavior_signals.py`: yawn, looking-away, head-down.
- `mddad/realtime_event_manager.py`: temporal confirmation.
- `mddad/realtime_monitor.py`: camera, dispatch, SocketIO, DB, evidence.
- `mddad/sockets.py`: start/stop realtime và audio chunks.
- `mddad/upload_routes.py`: upload và session detail.
- `models/driver_safety_v1/manifest.json`: contract Driver Safety v1.
- `models/openvino_driver_action_adas_0002/README.md`: contract action model.
