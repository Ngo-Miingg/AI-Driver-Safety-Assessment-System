# Handoff cho chat mới - AI Driver Safety Assessment System

## 1. Thông tin dự án

Project root:

```text
D:\Work\HocKy3\ComputerVision\MDDAD
```

Ứng dụng:

```text
AI Driver Safety Assessment System
```

Tài khoản demo:

```text
Username: admin
Password: admin123
```

Chạy project:

```powershell
cd D:\Work\HocKy3\ComputerVision\MDDAD
python app.py
```

URL local:

```text
http://127.0.0.1:5000
```

## 2. Trạng thái hiện tại

Hệ thống hiện đã có:

- Flask web app.
- Login/logout.
- Multi-driver management.
- SQLite persistence.
- Dashboard với dữ liệu thật.
- Upload video analysis.
- OpenCV metadata extraction.
- MediaPipe drowsiness/distraction analysis.
- Phone detection cho uploaded video.
- Realtime webcam monitoring bằng Flask-SocketIO + OpenCV + MediaPipe.
- Low-rate JPEG snapshot preview.
- Event evidence snapshots.
- Risk Score, Safety Score, Alert Level.
- Fatigue trend forecast.
- Event timeline.
- Rule-based AI driving report.
- History.
- Reports.
- Settings/calibration.
- Evaluation page.
- CSV export.
- Phone Test UI riêng để test ảnh/video với bounding box.
- Optional realtime phone detection với lựa chọn model theo từng phiên.

Định hướng mới của project:

```text
Mục tiêu cuối cùng không phải rule/threshold + detector riêng lẻ.
Hệ thống cần chuyển sang MULTIMODAL MODEL:
- input chính là video clip, không phải ảnh đơn
- có audio để nhận diện nói chuyện/cuộc gọi
- có skeleton/landmark sequence để học hành vi theo thời gian
- model tự nhận diện hành vi, bao gồm phone usage/phone call
- không để người dùng chỉnh ngưỡng thủ công cho quyết định hành vi
```

Tài liệu kiến trúc mới:

```text
docs/multimodal_model_plan.md
```

## 3. Các model phone detection hiện có

File model đang nằm trong:

```text
D:\Work\HocKy3\ComputerVision\MDDAD\models
```

Danh sách:

```text
models/phone_rtdetr_l.pt
models/phone_rtdetr_l_light_compare.pt
models/phone_yolo26n.pt
models/phone_yolo11n.pt
```

Vai trò:

| File | Vai trò |
|---|---|
| `phone_rtdetr_l.pt` | Model chính RT-DETR-L serious fine-tuned |
| `phone_rtdetr_l_light_compare.pt` | RT-DETR-L comparison checkpoint |
| `phone_yolo26n.pt` | YOLO26n fine-tuned, baseline/fallback nhẹ |
| `phone_yolo11n.pt` | Compatibility copy cho workflow cũ |

Tài liệu model:

```text
models/phone_detection/README.md
```

## 4. Kết quả train/fine-tune hiện tại

Dataset phone:

```text
Roboflow Mobile-phone v7
Format: YOLO detection format
Class: Mobile-phone
```

Dataset gốc:

| Split | Images | Label files | Bounding boxes |
|---|---:|---:|---:|
| Train | 9009 | 9009 | 12716 |
| Validation | 326 | 326 | 451 |
| Test | 185 | 185 | 265 |

Bản train chính RT-DETR-L serious:

| Split | Images | Label files | Bounding boxes |
|---|---:|---:|---:|
| Train | 4500 | 4500 | 6368 |
| Validation | 326 | 326 | 451 |
| Test | 185 | 185 | 265 |

Kết quả so sánh model:

| Model | Train images | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n baseline | 3000 | 21 | 0.93237 | 0.86031 | 0.92854 | 0.76828 | 5.1 MB |
| RT-DETR-L compare | 3000 | 14 | 0.93969 | 0.93286 | 0.95693 | 0.77543 | 63.1 MB |
| RT-DETR-L serious | 4500 | 8 | 0.93331 | 0.92239 | 0.95497 | 0.80077 | 63.1 MB |

Quyết định:

```text
RT-DETR-L serious được chọn làm model chính vì mAP50-95 cao nhất.
YOLO26n được giữ làm baseline/fallback vì nhẹ và nhanh hơn.
```

## 5. Thay đổi mới nhất: chọn model khi bắt đầu realtime

Trang:

```text
/realtime
```

Đã thêm khối **Model Selection** trước khi bấm Start Monitoring.

### Behavior recognition model

Hiện tại chỉ có:

```text
Built-in MediaPipe/OpenCV rules
```

Ý nghĩa:

- Đây là pipeline hiện tại.
- Dùng MediaPipe Face Mesh.
- Tính EAR cho mắt nhắm.
- Tính MAR cho ngáp.
- Head pose/looking away/head down bằng heuristic.
- Temporal thresholds từ Settings.
- Chưa có model action recognition train riêng.

### Phone detector model

Các lựa chọn:

```text
Disabled
RT-DETR-L serious fine-tuned
RT-DETR-L comparison checkpoint
YOLO26n fine-tuned
```

Khi bấm Start Monitoring, browser gửi SocketIO payload:

```json
{
  "behavior_model_backend": "builtin_mediapipe",
  "phone_detector_backend": "disabled | rtdetr_finetuned | rtdetr_light_compare | yolo_finetuned"
}
```

Server validate payload trước khi chạy realtime.

## 6. Các file đã sửa/thêm gần đây

Model registry:

```text
mddad/model_options.py
```

Realtime UI:

```text
templates/realtime.html
static/js/realtime.js
static/css/main.css
```

Realtime backend:

```text
mddad/sockets.py
mddad/realtime_monitor.py
mddad/realtime_routes.py
```

Detector backend:

```text
mddad/ai/detectors/factory.py
mddad/ai/detectors/ultralytics_detector.py
```

Phone test:

```text
mddad/phone_test_routes.py
templates/phone_test.html
```

Config:

```text
config.py
.env.example
```

Docs:

```text
models/phone_detection/README.md
docs/CHAT_HANDOFF_NEW_CHAT.md
```

## 7. Config quan trọng

Trong `config.py`:

```python
PHONE_DETECTOR_BACKEND = "rtdetr_finetuned"
YOLO_FINETUNED_MODEL_PATH = models/phone_yolo26n.pt
RTDETR_MODEL_PATH = models/phone_rtdetr_l.pt
RTDETR_LIGHT_COMPARE_MODEL_PATH = models/phone_rtdetr_l_light_compare.pt
REALTIME_PHONE_DETECTOR_BACKEND = "yolo_finetuned"
```

Lưu ý:

- Full AI upload mặc định dùng backend chính `rtdetr_finetuned`.
- Realtime nếu không chọn trực tiếp thì fallback config là `yolo_finetuned`.
- Trên `/realtime`, lựa chọn dropdown theo từng phiên sẽ override config fallback.

## 8. Các luồng chính

### 8.1. Realtime webcam flow

```text
User login
-> Select active driver
-> /realtime
-> Chọn behavior model
-> Chọn phone detector model
-> Start Monitoring
-> Browser emit start_realtime_session
-> Server validate payload
-> OpenCV mở webcam
-> MediaPipe xử lý frame
-> Optional low-rate phone detector xử lý frame
-> Emit realtime_update
-> Emit realtime_snapshot
-> Emit realtime_event nếu có event
-> Stop Monitoring
-> Save session, events, risk logs, report
-> Open saved session detail
```

### 8.2. Phone Test flow

```text
/phone-test
-> Upload image/video
-> Chọn 1 trong 3 model đã train
-> Run Detect Phone
-> Output có bounding box
```

Hiện `/phone-test` chỉ hiện 3 model đã train:

```text
RT-DETR-L serious fine-tuned
RT-DETR-L comparison checkpoint
YOLO26n fine-tuned
```

Không còn ưu tiên `YOLO pretrained` trong UI chính.

### 8.3. Uploaded video Full AI flow

```text
Upload video
-> OpenCV metadata/sample frames
-> MediaPipe face behavior analysis
-> Phone detector backend từ config
-> Temporal logic tạo events
-> Risk engine tính điểm
-> Save SQLite
-> Session detail/report/evaluation
```

## 9. Các kiểm tra đã chạy

Compile:

```powershell
python -m compileall mddad config.py app.py
```

Kết quả:

```text
OK
```

Render route bằng Flask test client:

```text
/dashboard OK
/drivers OK
/realtime OK
/phone-test OK
/upload OK
/history OK
/reports OK
/evaluation OK
/settings OK
```

Realtime page khi có active driver đã hiện đủ:

```text
behavior-model-select
phone-detector-select
Built-in MediaPipe/OpenCV rules
RT-DETR-L serious fine-tuned
RT-DETR-L comparison checkpoint
YOLO26n fine-tuned
Disabled
```

SocketIO test:

- `start_realtime_session` nhận payload hợp lệ.
- Payload được truyền xuống `realtime_manager.start`.
- Payload không hợp lệ bị reject bằng `realtime_error`.
- Các backend sau đều được server chấp nhận:

```text
disabled
rtdetr_finetuned
rtdetr_light_compare
yolo_finetuned
```

Camera debug:

```json
{
  "configured_index": 0,
  "results": [
    {
      "frame_read": true,
      "frame_shape": [480, 640, 3],
      "index": 0,
      "opened": true
    }
  ]
}
```

Realtime thật với webcam và phone detector Disabled:

```text
realtime_update: OK
realtime_snapshot: OK
realtime_completed: OK
session detail after stop: OK
```

Detector load/inference sanity check:

```text
rtdetr_finetuned: load OK, inference OK
rtdetr_light_compare: load OK, inference OK
yolo_finetuned: load OK, inference OK
```

Lưu ý:

```text
RT-DETR test trên frame đen giả lập có thể trả detection, nhưng mục tiêu test đó chỉ là kiểm tra model load/inference không lỗi.
```

## 10. Điểm cần chú ý khi tiếp tục

### 10.1. Naming còn sót YOLO

Trong code vẫn còn một số tên config/key lịch sử như:

```text
YOLO_CONFIDENCE_THRESHOLD
REALTIME_ENABLE_YOLO
REALTIME_YOLO_SAMPLE_FPS
analysis_mode = realtime_mediapipe_yolo
```

Lý do giữ:

```text
Tránh migration lớn và tránh phá schema/settings hiện tại.
```

Hướng nên làm sau:

```text
Đổi dần wording UI/report sang "phone detector" thay vì "YOLO".
Không đổi DB/schema vội nếu không cần.
```

### 10.2. Behavior model

Hiện chưa có model train riêng cho hành vi bất thường.

Hiện tại đang dùng:

```text
MediaPipe/OpenCV rules
```

Hướng tiếp theo:

```text
Train action recognition model cho:
- normal
- eye closure/drowsy
- yawning
- looking away
- head down
- phone usage
- mixed-risk
```

Dataset định hướng:

```text
AI City Track 3 / M2DAR-style dataset
```

Yêu cầu mới:

```text
Behavior model phải tiến tới multimodal video-audio-skeleton model.
MediaPipe chỉ nên đóng vai trò feature extractor cho skeleton/landmark, không phải rule engine cuối cùng.
Phone detector YOLO/RT-DETR chỉ nên là baseline/teacher/auxiliary feature, không phải model hành vi chính.
```

## 11. Slide/báo cáo đang làm

Tài liệu nguồn slide:

```text
docs/AI_Driver_Safety_Slide_Source_Brief_v2_papers.docx
```

Prompt tạo slide:

```text
docs/slide_progress_assets/06_prompt/PROMPT_TAO_SLIDE_TIEN_DO.md
```

Assets slide:

```text
docs/slide_progress_assets
```

Ảnh ghép YOLO vs RT-DETR results:

```text
docs/slide_progress_assets/07_slide_specific/yolo26n_vs_rtdetrl_results_side_by_side.png
```

Nội dung trình bày chính:

```text
Bước 1: So sánh YOLO26n và RT-DETR-L.
Bước 2: RT-DETR-L có Recall/mAP tốt hơn nên chọn fine-tune sâu hơn.
Bước 3: Fine-tune RT-DETR-L serious đạt mAP50-95 cao nhất.
Bước 4: Đưa model vào project và test qua /phone-test + realtime optional detector.
```

## 12. Việc nên làm tiếp

Ưu tiên kỹ thuật:

0. Chốt và triển khai hướng multimodal model theo `docs/multimodal_model_plan.md`:
   - video clip input
   - audio speech/call input
   - skeleton/landmark sequence input
   - model tự phân loại hành vi, kể cả phone usage/phone call
   - không dựa vào ngưỡng thủ công làm quyết định chính

1. Chạy thủ công `/realtime` với từng phone detector nếu cần baseline:
   - Disabled
   - YOLO26n
   - RT-DETR-L serious
   - RT-DETR-L compare

2. Đo FPS thực tế:
   - Realtime MediaPipe-only.
   - Realtime + YOLO26n.
   - Realtime + RT-DETR-L.

3. Nếu RT-DETR-L realtime quá chậm:
   - Giữ RT-DETR-L cho upload/phone-test.
   - Dùng YOLO26n cho realtime fallback.

4. Dọn wording UI/report:
   - Dùng “phone detector” thay vì “YOLO” ở các text người dùng nhìn thấy.

5. Nếu có thời gian:
   - Lưu `behavior_model_backend` và `phone_detector_backend` vào session metadata/schema để session detail biết phiên đó đã chọn model nào.

Ưu tiên báo cáo/slide:

1. Slide kiến trúc multimodal video-audio-skeleton.
2. Slide baseline hiện tại: MediaPipe/OpenCV rules + phone detector.
3. Slide so sánh YOLO26n vs RT-DETR-L như phone auxiliary/baseline.
4. Slide lý do cần train/fine-tune video behavior model thay vì image-only detector.
5. Slide hướng tiếp theo: train multimodal behavior model và ablation RGB/skeleton/audio.

## 13. Câu mô tả ngắn cho chat mới

Có thể copy đoạn này vào chat mới:

```text
Tôi đang làm project AI Driver Safety Assessment System tại D:\Work\HocKy3\ComputerVision\MDDAD.
Hệ thống Flask đã có login, driver, dashboard, upload analysis, realtime webcam, snapshot preview, evidence snapshot, history, reports, settings, evaluation CSV.
Phone detector đã train/fine-tune 3 model: RT-DETR-L serious, RT-DETR-L compare, YOLO26n.
RT-DETR-L serious là model chính vì mAP50-95 cao nhất; YOLO26n là fallback nhẹ.
Gần nhất đã thêm dropdown chọn model trên /realtime:
- Behavior model: mặc định Built-in MediaPipe/OpenCV rules
- Phone detector: Disabled, RT-DETR-L serious, RT-DETR-L compare, YOLO26n
SocketIO start_realtime_session đã gửi behavior_model_backend và phone_detector_backend.
Đã kiểm compile, route render, socket payload, camera debug, realtime update/snapshot/completed.
Tiếp theo cần kiểm thủ công FPS từng model, hoàn thiện wording UI/report, và tiếp tục làm slide/báo cáo.
```

## 14. Bổ sung sau khi rà lại toàn bộ luồng

Phần này được thêm sau khi kiểm lại route/code để tránh sót tính năng khi chuyển sang chat mới.

### 14.1. Các route/tính năng đã có nhưng handoff ban đầu chưa nêu đủ

Auth/user:

```text
/login
/register
/logout
/language/<language>
```

Ghi chú:

- Có đăng nhập/đăng xuất.
- Có đăng ký tài khoản local.
- Có chuyển ngôn ngữ Anh/Việt.
- Demo account vẫn là `admin / admin123`.

Health/foundation:

```text
/
/foundation
/health
```

Ghi chú:

- `/health` dùng kiểm tra app còn chạy.
- `/foundation` là trang trạng thái nền tảng ban đầu.

Driver:

```text
/drivers
/drivers/new
/drivers/<driver_id>
/drivers/<driver_id>/edit
/drivers/<driver_id>/select
```

Ghi chú:

- Có tạo, xem chi tiết, sửa và chọn active driver.
- Chưa có route xóa driver riêng.
- Realtime/upload cần active driver.

Upload/session:

```text
/upload
/sessions/<session_id>
/sessions/<session_id>/analyze
/sessions/<session_id>/simulate-analysis
/sessions/<session_id>/mediapipe-analysis
/sessions/<session_id>/full-ai-analysis
/evidence/<filename>
```

Ghi chú:

- Upload tạo session video.
- Session detail có các nút phân tích.
- Simulated analysis vẫn còn.
- MediaPipe analysis vẫn còn.
- Full AI analysis vẫn còn.
- `/evidence/<filename>` phục vụ ảnh minh chứng sự kiện.

Realtime:

```text
/realtime
/debug/camera
SocketIO: start_realtime_session
SocketIO: stop_realtime_session
```

Ghi chú:

- `/debug/camera` chỉ dùng khi debug.
- Đã test camera index 0 mở được.

Phone/behavior test:

```text
/phone-test
/phone-test/output/<filename>
/behavior-test
/behavior-test/output/<filename>
```

Ghi chú:

- `/phone-test` test phone detector trên ảnh/video, output có bounding box.
- `/behavior-test` là giao diện test behavior recognition riêng, phục vụ hướng action recognition/VITDD.
- Handoff ban đầu đã nói `/phone-test` nhưng chưa nhắc rõ `/behavior-test`.

Dashboard/history/report/evaluation/settings:

```text
/dashboard
/history
/reports
/evaluation
/evaluation.csv
/settings
```

Ghi chú:

- Evaluation CSV đã có.
- Settings vẫn dùng các key lịch sử có chữ YOLO cho một số threshold, nhưng UI đang được dọn dần sang wording "phone detector".

### 14.2. Luồng Behavior Test

Luồng này tồn tại riêng với `/behavior-test`.

```text
User login
-> /behavior-test
-> Upload image/video
-> Chọn hoặc dùng cấu hình behavior model hiện có
-> Hệ thống chạy behavior recognition test
-> Output hiển thị kết quả dự đoán/hành vi
```

Tình trạng:

```text
Đây là khu vực thử nghiệm cho hướng behavior/action recognition.
Model hành vi train hoàn chỉnh chưa phải phần chính của workflow realtime.
Realtime hiện vẫn dùng Built-in MediaPipe/OpenCV rules.
```

Tài liệu liên quan:

```text
docs/vitdd_behavior_integration.md
kaggle/vitdd_driver_behavior_kaggle_train.ipynb
kaggle/README_KAGGLE_VITDD_BEHAVIOR_TRAIN.md
tools/prepare_vitdd_behavior_training.py
```

### 14.3. Luồng phân tích trong Session Detail

Một session video upload có thể đi qua nhiều mức phân tích:

```text
Upload video
-> Session detail
-> Run simulated analysis
   hoặc Run MediaPipe analysis
   hoặc Run Full AI Analysis
-> Clear previous results
-> Store events
-> Store risk logs
-> Store report
-> Update session metrics
```

Vai trò từng chế độ:

| Chế độ | Vai trò |
|---|---|
| Simulated analysis | Giữ lại để demo scoring/report khi chưa cần AI thật |
| MediaPipe analysis | Phân tích mặt, mắt, miệng, head pose bằng landmark/rule |
| Full AI Analysis | MediaPipe + phone detector đã train/fine-tune |

Điểm cần nhớ:

```text
Không được xóa simulated analysis.
Không được xóa upload analysis.
Không được xóa Full AI Analysis.
```

### 14.4. Luồng evidence snapshot

Realtime có thể lưu ảnh minh chứng sự kiện:

```text
Realtime event confirmed
-> Save small JPEG under uploads/evidence/
-> Store relative evidence_image_path in event record
-> Session detail displays evidence thumbnail
```

Route xem ảnh:

```text
/evidence/<filename>
```

Giới hạn:

```text
Không lưu full video stream.
Chỉ lưu ảnh JPEG nhỏ theo sự kiện.
```

### 14.5. Luồng ngôn ngữ

Hệ thống có language switch:

```text
/language/en
/language/vi
```

Tình trạng:

```text
Có hỗ trợ chuyển Anh/Việt qua i18n.py.
Một số text mới hoặc text kỹ thuật có thể vẫn cần rà tiếp để Việt hóa hoàn toàn.
```

### 14.6. Notebook/dataset/tooling cần nhớ

Dataset phone hiện có:

```text
dataset/roboflow_mobilephone_wusj2_v7/extracted
dataset/kaggle_upload/*.zip
```

Notebook quan trọng:

```text
kaggle/phone_detection_combined_light_comparison_kaggle.ipynb
kaggle/phone_detection_rtdetr_serious_4h_finetune_kaggle.ipynb
kaggle/phone_detection_yolo26n_light_kaggle_train.ipynb
kaggle/vitdd_driver_behavior_kaggle_train.ipynb
```

Tool scripts quan trọng:

```text
tools/download_roboflow_mobilephone.py
tools/prepare_kaggle_combined_light_comparison.py
tools/prepare_kaggle_rtdetr_phone_serious_4h_finetune.py
tools/prepare_vitdd_behavior_training.py
tools/build_slide_source_doc.py
```

Kết quả train/eval đã extract:

```text
D:\Work\output_extracted\test_yolo_rtdetr
D:\Work\output_extracted\finetune_rtdetr
```

### 14.7. Tài liệu quan trọng khác

Ngoài handoff này, các file nên giữ trong ngữ cảnh:

```text
docs/CODEX_HANDOFF.md
docs/phone_detector_training_report.md
docs/offline_deployment.md
docs/demo_script.md
docs/testing_checklist.md
docs/demo_assets_guide.md
docs/evaluation_plan.md
docs/model_guide.md
docs/presentation_script.md
docs/slide_review_report.md
docs/AI_Driver_Safety_Slide_Source_Brief_v2_papers.docx
docs/Slide.pptx
```

### 14.8. Điểm vừa phát hiện trong lúc kiểm luồng

Một số text người dùng nhìn thấy vẫn còn chữ `YOLO` dù backend mặc định hiện đã chuyển sang phone detector/RT-DETR:

```text
settings labels
report text
session detail warning labels
translation strings
analysis_mode = realtime_mediapipe_yolo
```

Đã bắt đầu sửa một phần wording sang `phone detector`, nhưng chưa nên đổi toàn bộ key/schema ngay vì:

```text
REALTIME_ENABLE_YOLO, YOLO_CONFIDENCE_THRESHOLD, realtime_mediapipe_yolo là key lịch sử.
Đổi mạnh có thể ảnh hưởng settings/schema/report cũ.
```

Khuyến nghị:

```text
Giữ key kỹ thuật cũ nếu chưa cần migration.
Chỉ đổi wording UI/report sang "phone detector" để demo không bị sai khi chọn RT-DETR.
```

### 14.9. Kiểm tra bổ sung đã chạy sau khi rà luồng

Đã kiểm:

```text
python -m compileall mddad config.py app.py
```

Đã kiểm route:

```text
/dashboard
/drivers
/realtime
/phone-test
/upload
/history
/reports
/evaluation
/settings
```

Đã kiểm `/realtime` khi có active driver:

```text
behavior-model-select: có
phone-detector-select: có
Disabled: có
RT-DETR-L serious fine-tuned: có
RT-DETR-L comparison checkpoint: có
YOLO26n fine-tuned: có
```

Đã kiểm SocketIO:

```text
payload hợp lệ được truyền xuống realtime_manager.start
payload sai behavior model bị reject
payload sai phone detector bị reject
disabled/rtdetr_finetuned/rtdetr_light_compare/yolo_finetuned đều được chấp nhận
```

Đã kiểm camera:

```text
/debug/camera opened=true, frame_read=true, shape=[480,640,3]
```

Đã kiểm realtime thật với webcam:

```text
Start realtime với phone detector Disabled
-> realtime_update OK
-> realtime_snapshot OK
-> stop OK
-> realtime_completed OK
-> session detail mở được
```

Đã kiểm detector load/inference:

```text
rtdetr_finetuned OK
rtdetr_light_compare OK
yolo_finetuned OK
```

Lưu ý:

```text
Kiểm detector chỉ là sanity check trên frame giả; cần test thủ công bằng ảnh/video cabin thật để đánh giá chất lượng.
```

### 14.10. Việc nên yêu cầu chat mới làm đầu tiên

Khi mở chat mới, nên yêu cầu:

```text
Đọc docs/CHAT_HANDOFF_NEW_CHAT.md, sau đó kiểm tiếp các luồng còn lại:
1. Full AI upload với backend RT-DETR.
2. Phone-test với cả 3 model trên ảnh/video thật.
3. Realtime với YOLO26n và RT-DETR-L để đo FPS.
4. Rà wording UI/report còn sót chữ YOLO khi đang dùng RT-DETR.
5. Nếu cần, lưu behavior_model_backend và phone_detector_backend vào session metadata/schema.
```

### 14.11. Kết luận rà soát tính năng/luồng

Sau khi đối chiếu lại route, schema, config, template và luồng realtime, handoff hiện đã đủ để chuyển sang chat mới. Những phần dễ bị sót đã được ghi bổ sung ở mục 14.

Checklist trạng thái:

| Nhóm | Tình trạng | Ghi chú |
|---|---|---|
| Auth | Đã có | Login, register, logout, demo account |
| Ngôn ngữ | Đã có | Anh/Việt qua `/language/<language>`; cần rà tiếp text mới nếu muốn Việt hóa tuyệt đối |
| Driver | Đã có phần chính | Tạo, xem, sửa, chọn active driver; chưa có xóa driver |
| Dashboard/history/reports | Đã có | Dùng dữ liệu SQLite thật |
| Upload analysis | Đã có | Simulated, MediaPipe, Full AI đều còn |
| Phone test riêng | Đã có | Chọn 3 model đã train, output bounding box |
| Behavior test riêng | Đã có | Hướng thử nghiệm action recognition/VITDD, chưa phải workflow chính realtime |
| Realtime webcam | Đã có | SocketIO, OpenCV camera, MediaPipe, snapshot preview, stop/save session |
| Chọn model realtime | Đã có | Behavior mặc định builtin; phone detector chọn Disabled/RT-DETR serious/RT-DETR compare/YOLO26n |
| Evidence snapshot | Đã có | Lưu JPEG nhỏ trong `uploads/evidence/`, hiển thị ở session detail nếu có |
| Settings/calibration | Đã có | Threshold, FPS, snapshot, detector settings; một số key lịch sử vẫn tên YOLO |
| Evaluation CSV | Đã có | `/evaluation.csv` |
| Model files | Đã có | RT-DETR-L serious, RT-DETR-L compare, YOLO26n, compatibility YOLO11n |
| Training artifacts | Đã có | Notebook, dataset zip, chart/table/report đã ghi trong handoff |
| Slide/báo cáo | Đã có nhiều tài liệu | Cần tiếp tục chọn lọc để đưa vào slide cuối |

Các phần chưa hoàn thiện nhưng đã được ghi rõ:

```text
1. Chưa có model action recognition/multimodal hoàn chỉnh thay thế MediaPipe/OpenCV rules.
2. Chưa lưu behavior_model_backend và phone_detector_backend vào schema session, hiện chủ yếu dùng theo phiên runtime.
3. Realtime RT-DETR-L cần test FPS thực tế; nếu chậm thì giữ RT-DETR cho upload/phone-test và dùng YOLO26n realtime fallback.
4. Một số wording/key kỹ thuật còn chữ YOLO vì lý do tương thích lịch sử.
5. Cần test thủ công bằng video cabin thật cho phone detection, drowsiness, yawn, looking away, head down.
```

Các luồng chính đã được mô tả đủ:

```text
Login -> chọn driver -> realtime webcam -> realtime_update/snapshot/event -> stop -> session detail/report/evaluation
Login -> chọn driver -> upload video -> metadata -> simulated/MediaPipe/Full AI -> session detail/report/evaluation
Phone-test -> chọn ảnh/video -> chọn model phone detector -> output bounding box
Behavior-test -> chọn ảnh/video -> model hành vi thử nghiệm -> output hành vi
Settings -> chỉnh threshold/FPS/detector config -> áp dụng cho phiên/phân tích mới
Evaluation -> xem chỉ số phiên -> export CSV
```

## 15. Cập nhật kiến trúc mới: đã đập luồng cũ khỏi app chính

Theo yêu cầu mới, workflow chính đã chuyển sang:

```text
Login
-> chọn driver
-> upload video
-> đọc metadata nếu cần
-> Run Multimodal Analysis
-> model video-audio-skeleton sinh behavior events
-> risk/report/evaluation
```

Các route/UI legacy đã bị tháo khỏi app startup:

```text
/realtime
/phone-test
/behavior-test
/settings
/sessions/<id>/simulate-analysis
/sessions/<id>/mediapipe-analysis
/sessions/<id>/full-ai-analysis
```

Các file UI/controller legacy đã xóa:

```text
templates/realtime.html
templates/phone_test.html
templates/behavior_test.html
templates/settings.html
static/js/realtime.js
mddad/realtime_routes.py
mddad/realtime_monitor.py
mddad/sockets.py
mddad/phone_test_routes.py
mddad/behavior_test_routes.py
mddad/model_options.py
```

Route phân tích AI hiện còn lại:

```text
/sessions/<session_id>/multimodal-analysis
```

Model contract mới:

```text
MULTIMODAL_MODEL_PATH=models/multimodal_driver_behavior.pt
MULTIMODAL_MODEL_BACKEND=video_audio_skeleton
Input: RGB video clip windows + audio signal + skeleton/landmark sequence
Output: behavior classes/probabilities, including phone_usage and phone_call
```

Lưu ý quan trọng:

```text
Chưa được fallback sang threshold/rule cũ.
Nếu thiếu weights multimodal thật, endpoint sẽ báo thiếu model thay vì giả lập kết quả bằng MediaPipe/YOLO.
```
