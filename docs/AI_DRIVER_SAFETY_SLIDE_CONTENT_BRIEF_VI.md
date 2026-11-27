# TÀI LIỆU NỘI DUNG TẠO SLIDE

## PHÁT HIỆN BUỒN NGỦ VÀ GIẢM CHÚ Ý CỦA TÀI XẾ BẰNG MÔ HÌNH ĐA PHƯƠNG THỨC TRONG CABIN

### 1. Mục tiêu tài liệu

Tài liệu này dùng làm nguồn duy nhất cho AI thiết kế slide giới thiệu dự án. Nội dung phải giữ đúng những gì hệ thống hiện có, phân biệt rõ model, bộ trích xuất đặc trưng, luật xác nhận sự kiện và giao diện ứng dụng. Không tự thêm số liệu, class, model hoặc chức năng chưa được kiểm chứng.

Thông tin nhóm:

- Ngo Van Minh, Nguyen Thi Phuong
- Group 01
- Dai Nam University
- Hanoi, Vietnam

### 2. Thông điệp trung tâm

Tai nạn trong cabin thường bắt đầu từ các dấu hiệu ngắn và khó quan sát liên tục: mắt nhắm, buồn ngủ, ngáp, nhìn lệch, cúi đầu, tương tác với điện thoại hoặc thay đổi tư thế. Dự án xây dựng một hệ thống giám sát tài xế theo thời gian thực, kết hợp nhận diện khuôn mặt, suy luận theo chuỗi thời gian, nhận diện hành động trong clip và lớp xác nhận sự kiện để chuyển tín hiệu hình ảnh thành cảnh báo và báo cáo có thể theo dõi.

Thông điệp ngắn cho slide mở đầu:

> Không chỉ nhìn một khung hình; hệ thống theo dõi diễn biến hành vi của tài xế theo thời gian để đưa ra cảnh báo có căn cứ hơn.

### 3. Trạng thái sản phẩm hiện tại

Pipeline đang được ưu tiên cho demo chính là `Driver Safety v1`:

```text
Camera frame
  -> face detection
  -> face verification / identity status
  -> facial feature extraction
  -> temporal drowsiness model
  -> OpenVINO action recognition on a 16-frame clip
  -> temporal smoothing and event confirmation
  -> risk score, safety score, alert level
  -> realtime UI, event log, saved session and report
```

Phone detector độc lập đang tạm tắt trong realtime để tránh trộn tín hiệu với model hành vi. Nhắn tin và gọi điện trong demo chính phải được mô tả là kết quả của action model.

Audio và skeleton đa phương thức không được mô tả là đang chạy trong `Driver Safety v1` nếu phiên demo không bật đúng pipeline tương ứng. Có thể đưa chúng vào slide “định hướng multimodal” hoặc slide về các artifact nghiên cứu đã chuẩn bị, nhưng phải ghi rõ trạng thái.

### 4. Các model và thành phần đang sử dụng

#### 4.1. Face detector

- Artifact: `models/driver_safety_v1/face_detection.pt`.
- Vai trò: định vị khuôn mặt tài xế trong frame.
- Output: vùng khuôn mặt, trạng thái có/không có mặt.
- Không dùng để phân loại toàn bộ hành vi.

#### 4.2. Face verification

- Artifact: `models/driver_safety_v1/face_verification.h5`.
- Vai trò: xác thực tài xế đã đăng ký.
- Workflow: đăng ký nhiều góc mặt, trích embedding, lưu mẫu đại diện, so sánh embedding khi realtime.
- Output: verified, mismatch, not enrolled hoặc verifier unavailable.
- Đây là lớp nhận dạng danh tính, không phải lớp nhận diện buồn ngủ.

#### 4.3. Drowsiness model

- Artifact: `models/driver_safety_v1/best_model.h5`.
- Scaler: `models/driver_safety_v1/scaler.joblib`.
- Sáu đặc trưng đầu vào: `left_ear`, `right_ear`, `mar`, `pitch`, `yaw`, `roll`.
- Hệ thống gom chuỗi 15 frame trước khi suy luận.
- Output chính: drowsiness probability và drowsiness state.
- Event không được tạo chỉ từ một frame; phải qua temporal confirmation.

#### 4.4. OpenVINO driver action recognition

- Model: `driver-action-recognition-adas-0002`.
- Artifact runtime: encoder và decoder OpenVINO trong `models/openvino_driver_action_adas_0002/FP16-INT8`.
- Input: clip 16 frame, lấy mẫu theo stride cấu hình.
- Vai trò: nhận diện hành động của tài xế theo ngữ cảnh chuyển động, không phải detector ảnh tĩnh.
- Các lớp công khai trên sản phẩm:
  - Lái xe bình thường
  - Nhắn tin
  - Gọi điện
  - Điều khiển radio
  - Ăn uống
  - Với tay ra sau
  - Chỉnh tóc/trang điểm
- Các biến thể hướng nội bộ của checkpoint đã được gộp thành một nhãn logic trước khi gửi ra UI.
- Hệ thống dùng smoothing và confirmation để tránh tạo sự kiện từ dự đoán thoáng qua.
- Lớp Ăn uống đang dùng ngưỡng chấp nhận riêng cao hơn vì có xu hướng nhầm với tư thế bình thường.

#### 4.5. MediaPipe Face Mesh và OpenCV

- OpenCV: đọc camera, resize frame, encode JPEG, tạo stream, vẽ overlay và lưu snapshot.
- MediaPipe Face Mesh: trích landmark khuôn mặt làm feature extractor.
- Từ landmark có thể tạo các tín hiệu độc lập: ngáp, nhìn lệch và cúi đầu.
- Các tín hiệu này được xác nhận theo thời gian trước khi tạo event.
- MediaPipe không thay thế action model và không tự nhận diện các lớp action của OpenVINO.

#### 4.6. Phone detector độc lập

- RT-DETR/YOLO phone detector vẫn tồn tại trong thư mục model để test riêng.
- Trạng thái hiện tại: tạm tắt khỏi realtime chính.
- Không đưa detector độc lập vào sơ đồ pipeline demo chính.
- Nếu trình bày, đặt ở slide “thành phần dự phòng / hướng mở rộng”, không gọi nó là output của action model.

### 5. Chức năng sản phẩm

#### Người dùng và tài xế

- Đăng nhập, đăng ký, đăng xuất.
- Chuyển ngôn ngữ Anh/Việt.
- Tạo, sửa, xem và chọn tài xế đang hoạt động.
- Đăng ký xác thực khuôn mặt bằng nhiều góc nhìn.

#### Giám sát realtime

- Kiểm tra camera.
- Mở phiên giám sát.
- Hiển thị stream camera.
- Hiển thị trạng thái khuôn mặt và xác thực danh tính.
- Hiển thị xác suất buồn ngủ và trạng thái cảnh báo.
- Hiển thị hành động đang được model nhận diện.
- Hiển thị bộ đệm frame, thời lượng cửa sổ và số sự kiện.
- Ghi event sau khi tín hiệu được xác nhận theo thời gian.
- Lưu snapshot minh chứng cho event nếu cấu hình bật.
- Dừng phiên và lưu kết quả.

#### Quản lý kết quả

- Dashboard tổng quan.
- Lịch sử phiên lái.
- Chi tiết session.
- Báo cáo rủi ro và mức an toàn.
- Timeline sự kiện.
- Evaluation và xuất CSV.
- Health check cho app, database, camera và model readiness.

### 6. Luồng xử lý cần vẽ trong slide

Sơ đồ chính nên dùng bố cục ngang, ít chữ, các khối nối bằng mũi tên:

```text
[Camera frame]
      |
      v
[Face detector] ---> [Face verification]
      |
      +--> [Face landmarks: EAR / MAR / head pose]
      |                    |
      |                    v
      |            [Drowsiness temporal model]
      |
      +--> [16-frame action clip]
                           |
                           v
                 [OpenVINO action model]
                           |
                           v
              [Smoothing + event confirmation]
                           |
                           v
          [Risk / Safety / Alert / Event / Report]
```

Một sơ đồ phụ cho giao diện:

```text
Camera preview | Live state cards | Action probabilities
       \              |                 /
        \             v                /
             Realtime monitoring console
                         |
              Session / Timeline / Report
```

Không vẽ audio, skeleton hoặc multimodal fusion như một thành phần đang chạy trong realtime chính nếu không có ảnh chụp và log chứng minh tương ứng.

### 7. Cấu trúc slide đề xuất

#### Slide 1 - Tên dự án và thông điệp

Tiêu đề in hoa:

`PHÁT HIỆN BUỒN NGỦ VÀ GIẢM CHÚ Ý CỦA TÀI XẾ BẰNG MÔ HÌNH ĐA PHƯƠNG THỨC TRONG CABIN`

Phụ đề: Ngo Van Minh, Nguyen Thi Phuong - Group 01 - Dai Nam University.

Hình: một ảnh cabin/tài xế rõ chủ thể, không dùng ảnh nền quá tối hoặc ảnh stock không liên quan.

#### Slide 2 - Vấn đề thực tế

- Buồn ngủ và giảm chú ý thường xuất hiện trước khi xảy ra sự cố.
- Quan sát thủ công không liên tục và khó mở rộng cho nhiều tài xế.
- Một frame đơn lẻ dễ tạo cảnh báo sai.
- Cần theo dõi hành vi theo chuỗi thời gian và lưu lại bằng chứng.

Hình: minh họa tài xế trong cabin cùng các tín hiệu mắt, miệng, hướng nhìn và điện thoại.

#### Slide 3 - Mục tiêu và giải pháp

- Theo dõi tài xế qua camera.
- Xác thực đúng tài xế đang được giám sát.
- Nhận diện buồn ngủ và hành vi gây mất tập trung.
- Xác nhận sự kiện theo thời gian.
- Chuyển output model thành cảnh báo, timeline và báo cáo.

#### Slide 4 - Kiến trúc hệ thống

Dùng sơ đồ ở mục 6. Nhấn mạnh sự khác nhau giữa feature extraction, model inference, temporal confirmation và application reporting.

#### Slide 5 - Các model chính

Trình bày theo 4 cột: thành phần, input, output, vai trò.

| Thành phần | Input | Output | Vai trò |
|---|---|---|---|
| Face detector | Frame camera | Face box | Tìm khuôn mặt |
| Face verification | Nhiều góc mặt | Identity status | Xác thực tài xế |
| Drowsiness model | Chuỗi đặc trưng khuôn mặt | Probability | Ước lượng buồn ngủ |
| OpenVINO action model | Clip 16 frame | Hành vi | Nhận diện nhắn tin/gọi điện và các action |
| MediaPipe Face Mesh | Face crop | Landmarks | Tạo EAR, MAR, hướng nhìn, cúi đầu |
| OpenCV | Camera/frame | Stream/overlay | Vận hành realtime |

#### Slide 6 - Dữ liệu và đầu vào

- Frame camera được xử lý theo thời gian.
- Drowsiness dùng 6 đặc trưng khuôn mặt.
- Action model dùng clip 16 frame.
- Kết quả chỉ được xác nhận sau khi đủ buffer và đạt điều kiện ổn định.
- Face enrollment dùng nhiều góc nhìn để tăng độ tin cậy xác thực.

Không ghi số lượng dataset hoặc accuracy nếu không có bảng nghiệm thu tương ứng.

#### Slide 7 - Nhận diện hành vi và temporal reasoning

Minh họa ba tầng:

1. Instant prediction: model vừa dự đoán gì.
2. Stable prediction: nhãn lặp lại đủ lâu và đạt ngưỡng.
3. Confirmed event: event được ghi vào timeline sau thời gian xác nhận.

Thông điệp: hệ thống không biến một frame nhiễu thành cảnh báo chính thức.

#### Slide 8 - Giao diện realtime

Chèn Ảnh giao diện 1: màn hình `/realtime`.

Chú thích cần thể hiện:

- Live camera.
- Kết nối và trạng thái phiên.
- Khuôn mặt / xác thực.
- Xác suất buồn ngủ.
- Hành vi đang nhận diện.
- Xác suất các lớp logic.
- Bộ đệm hành vi.
- Timeline event.

#### Slide 9 - Dashboard và quản lý phiên

Chèn Ảnh giao diện 2: dashboard.

Chú thích cần thể hiện:

- Tài xế đang chọn.
- Tổng số phiên.
- Safety score và risk score.
- Số event theo loại.
- Truy cập lịch sử, báo cáo và evaluation.

#### Slide 10 - Session detail / report

Chèn Ảnh giao diện 3: chi tiết phiên hoặc báo cáo.

Chú thích cần thể hiện:

- Thông tin phiên.
- Timeline.
- Event evidence snapshot.
- Risk breakdown.
- Safety explanation.
- Dữ liệu dùng cho đánh giá hoặc xuất CSV.

#### Slide 11 - Demo và nghiệm thu

Kịch bản demo ngắn:

1. Đăng nhập và chọn tài xế.
2. Mở realtime.
3. Kiểm tra camera.
4. Bắt đầu giám sát.
5. Giữ trạng thái bình thường để cho buffer ổn định.
6. Thực hiện hành vi nhắn tin hoặc gọi điện trong một clip đủ dài.
7. Quan sát raw probability, stable label và confirmed event.
8. Dừng phiên, mở session detail và kiểm tra timeline.

Không trình diễn bằng cách giữ một ảnh tĩnh trước camera; action model cần chuỗi frame có chuyển động và ngữ cảnh cabin.

#### Slide 12 - Kết quả, giới hạn và hướng phát triển

Kết quả hiện có:

- Có ứng dụng web end-to-end cho quản lý tài xế, realtime monitoring và báo cáo.
- Có face verification nhiều góc.
- Có temporal drowsiness inference.
- Có OpenVINO action recognition trên clip.
- Có smoothing, confirmation, event log và session persistence.
- Có thể chạy offline trên laptop khi đủ runtime/model.

Giới hạn cần nói trung thực:

- Action model phụ thuộc chất lượng góc quay, ánh sáng và ngữ cảnh cabin.
- Lớp Ăn uống có xu hướng nhầm với tư thế bình thường, nên đang dùng chính sách xác nhận bảo thủ hơn.
- Phone detector độc lập đang tạm tắt khỏi realtime chính.
- Audio/skeleton multimodal không được nhận là output của Driver Safety v1 nếu chưa bật pipeline tương ứng.

Hướng phát triển:

- Nghiệm thu trên video cabin có nhãn.
- Hiệu chỉnh theo camera và tư thế lái thực tế.
- Bổ sung benchmark latency/FPS và ma trận nhầm lẫn theo phiên bản.
- Tích hợp edge deployment ổn định trên laptop.
- Tách rõ model output, event logic và reporting layer trong giao diện.

### 8. Danh sách ảnh giao diện cần chụp và chèn

Tạo thư mục `docs/slide_assets/interface/` và đặt tên đúng:

1. `01_realtime_monitoring.png` - trang `/realtime`, có camera, trạng thái model, xác suất hành vi và timeline.
2. `02_dashboard_overview.png` - trang `/dashboard`, có KPI, tài xế đang chọn và phiên gần đây.
3. `03_session_detail_report.png` - trang `/sessions/<id>`, có event timeline, risk/safety score và evidence.
4. `04_face_enrollment.png` - trang đăng ký xác thực khuôn mặt nhiều góc.
5. `05_model_readiness.png` - màn hình hoặc phần hiển thị artifact/model readiness.

Quy tắc chụp:

- Dùng cùng một ngôn ngữ giao diện.
- Chụp toàn bộ vùng quan trọng trong một viewport 1366x768 hoặc 1440x900.
- Không để URL, tài khoản, đường dẫn máy cá nhân hoặc lỗi debug trong ảnh.
- Camera phải có khuôn mặt rõ và trạng thái model đã chạy.
- Không dùng ảnh placeholder màu đen làm ảnh sản phẩm.
- Mỗi ảnh cần một caption ngắn, không chèn quá nhiều chữ lên screenshot.

### 9. Prompt cho AI tạo slide

```text
Tạo một bộ slide tiếng Việt chuyên nghiệp để bảo vệ đồ án tốt nghiệp với chủ đề:
PHÁT HIỆN BUỒN NGỦ VÀ GIẢM CHÚ Ý CỦA TÀI XẾ BẰNG MÔ HÌNH ĐA PHƯƠNG THỨC TRONG CABIN.

Đối tượng người xem là giảng viên AI và hội đồng kỹ thuật. Hãy ưu tiên tính rõ ràng, bằng chứng và khả năng trình diễn thực tế. Không tự bịa số liệu, accuracy, dataset size, class hoặc tính năng chưa có trong tài liệu nguồn.

Sử dụng đúng cấu trúc 12 slide trong tài liệu nguồn: vấn đề, mục tiêu, kiến trúc, model, dữ liệu, temporal reasoning, giao diện realtime, dashboard, session report, demo, kết quả/giới hạn/hướng phát triển.

Phải phân biệt rõ:
- model inference;
- feature extraction bằng MediaPipe/OpenCV;
- temporal smoothing và event confirmation;
- risk/reporting layer của ứng dụng.

Pipeline demo chính là Driver Safety v1. Phone detector độc lập đang tạm tắt khỏi realtime. Nhắn tin và gọi điện phải được trình bày là output của OpenVINO action model. Không hiển thị hoặc nhắc đến nhãn theo hướng trái/phải; chỉ dùng các lớp logic: lái xe bình thường, nhắn tin, gọi điện, điều khiển radio, ăn uống, với tay ra sau, chỉnh tóc/trang điểm.

Chèn đúng các screenshot thật theo tên file trong mục 8. Nếu thiếu ảnh, tạo vùng giữ chỗ có nhãn rõ ràng thay vì tạo ảnh giao diện giả. Thiết kế có nền sáng, typography dễ đọc, màu xanh navy/cyan tiết chế, biểu đồ và sơ đồ ít chữ nhưng có luồng mũi tên rõ. Không dùng card dày đặc, không dùng gradient lớn, không làm dashboard giống landing page.

Mỗi slide có một thông điệp chính, tối đa 4-5 ý ngắn. Các slide kiến trúc và demo phải dùng hình ảnh/sơ đồ chiếm ưu tiên. Phần giới hạn phải giữ giọng học thuật, không che giấu các lớp có false positive cao. Kết thúc bằng giá trị thực tiễn: giám sát offline trên laptop, cảnh báo theo thời gian và lưu bằng chứng để đánh giá tài xế.
```

### 10. Nguyên tắc không được sai khi tạo slide

- Không gọi RT-DETR là model nhận diện gọi điện.
- Không gọi MediaPipe Face Mesh là model phân loại action.
- Không nói audio/skeleton đang chạy trong Driver Safety v1 nếu không có log/demo tương ứng.
- Không dùng kết quả của báo cáo nghiên cứu cũ để khẳng định trực tiếp cho realtime product hiện tại.
- Không ghi “model phát hiện điện thoại” trong slide chính khi phone detector đang disabled.
- Không biến xác suất tức thời thành event đã xác nhận.
- Không gộp các lớp khác nhau ngoài phạm vi đã thống nhất.
