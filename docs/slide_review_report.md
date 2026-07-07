# Slide Review Report

- File: `D:\Work\HocKy3\ComputerVision\MDDAD\docs\Slide.pptx`
- Slide count: **20**
- Slide size EMU: `18288000 x 10287000`
- File size: **8.67 MB**

## Findings

- **Potential blank slide:** [20]
- Font mix detected. Main fonts from runs: `Abadi` (601), `Times New Roman` (548), `+mj-lt` (18), `Arial` (15)
- XML font scan: `Times New Roman` (1639), `Abadi` (671), `MS Mincho` (249), `Arial` (110), `Arial Unicode Bold` (64), `+mj-lt` (21), `Calibri` (1)
- No extremely long single text shape detected by character count.

## Images Per Slide
- Slide 1: 1 image(s), 7 total shape(s)
- Slide 2: 2 image(s), 6 total shape(s)
- Slide 3: 3 image(s), 8 total shape(s)
- Slide 4: 3 image(s), 6 total shape(s)
- Slide 5: 2 image(s), 5 total shape(s)
- Slide 6: 2 image(s), 6 total shape(s)
- Slide 7: 2 image(s), 5 total shape(s)
- Slide 8: 2 image(s), 7 total shape(s)
- Slide 9: 3 image(s), 9 total shape(s)
- Slide 10: 2 image(s), 8 total shape(s)
- Slide 11: 2 image(s), 9 total shape(s)
- Slide 12: 2 image(s), 5 total shape(s)
- Slide 13: 2 image(s), 6 total shape(s)
- Slide 14: 3 image(s), 7 total shape(s)
- Slide 15: 3 image(s), 7 total shape(s)
- Slide 16: 3 image(s), 7 total shape(s)
- Slide 17: 2 image(s), 6 total shape(s)
- Slide 18: 2 image(s), 6 total shape(s)
- Slide 19: 2 image(s), 7 total shape(s)
- Slide 20: 0 image(s), 2 total shape(s)

## Slide Text Summary
### Slide 1
- AI Driver Safety Assessment System | Nhóm: Bug_Is_Feature | Sinh viên thực hiện:   Ngô Văn Minh                                    Hà Tuấn Huy |                                    Nguyễn Văn Nam
- Cuộc thi: Tài năng sáng tạo công nghệ
### Slide 2
- ĐẶT VẤN ĐỀ
- Buồn ngủ, mất tập trung và sử dụng điện thoại là những nguyên nhân hàng đầu gây tai nạn giao thông. | Các hành vi nguy hiểm như: | - Nhắm mắt kéo dài  | - Ngáp liên tục  | - Nhìn lệch khỏi đường  | - Cúi đầu  | - Sử dụng điện thoại  | Thường diễn ra trong thời gian ngắn nhưng lặp lại nhiều lần, rất khó giám sát bằng mắt thường. | Vì vậy, đề tài xây dựng hệ thống AI Driver Safety Assessment System sử dụng thị giác máy tính để phát hiện hành vi nguy hiểm, đánh giá mức độ rủi ro và hỗ trợ cảnh báo 
### Slide 3
- MỤC TIÊU
- Mục tiêu tổng quát của dự án là xây dựng một hệ thống phần mềm hoàn chỉnh từ đầu vào video/webcam đến đầu ra là đánh giá an toàn tài xế, gồm kết quả định lượng, timeline sự kiện và báo cáo giải thích.
- 1. Đầu vào: video upload hoặc webcam realtime.  | 2. Phân tích thị giác: OpenCV, MediaPipe Face Mesh, detector điện thoại.  | 3. Theo dõi hành vi: mắt nhắm, ngáp, nhìn lệch, cúi đầu, dùng điện thoại.  | 4. Đánh giá rủi ro: Temporal Logic, Risk Score, Safety Score, Alert Level.  | 5. Kết quả: dashboard, timeline sự kiện, báo cáo, lịch sử, CSV evaluation.
### Slide 4
- THÔNG TIN TRIỂN KHAI
### Slide 5
- LỘ TRÌNH TRIỂN KHAI
### Slide 6
- LỰA CHỌN PHƯƠNG PHÁP
- Dự án sử dụng mô hình thị giác máy tính đa nhánh gồm MediaPipe Face Mesh, Phone Detection, Temporal Event Tracking và Risk Engine. | MediaPipe Face Mesh: Phân tích khuôn mặt để phát hiện nhắm mắt, ngáp, nhìn lệch và cúi đầu thông qua các chỉ số EAR, MAR và Head Pose.  | Phone Detection: Sử dụng RT-DETR để phát hiện hành vi sử dụng điện thoại khi lái xe.  | Temporal Event Tracking: Theo dõi các hành vi theo thời gian, giảm nhiễu và hạn chế cảnh báo giả.  | Risk Engine: Tổng hợp các sự kiện để tín
### Slide 7
- XÂY DỰNG HỆ THỐNG
### Slide 8
- DATASET
- 1. Phone Detection Dataset | - Roboflow Mobile-phone v7 | - YOLO format, 1 class: Mobile-phone | - Dataset gốc: 9520 ảnh, 13432 bounding boxes | - Train chính: 4500 / 326 / 185 ảnh | - Bad labels: 0 | Vai trò: fine-tune detector phát hiện điện thoại | 2. Driver Behavior Dataset | - Định hướng: AI City Track 3 / M2DAR-style dataset | - Dữ liệu: RGB video cabin + nhãn hành vi theo thời gian | - Label: normal, drowsy, yawn, looking away, head down, phone usage | - Mục tiêu: train action recognition
### Slide 9
- LỰA CHỌN MODEL DETECT PHONE
- Mục tiêu: | - So sánh YOLO26n và RT-DETR-L trên cùng dataset Mobile-phone v7 | - Đánh giá khả năng phát hiện điện thoại trước khi chọn model fine-tune chính |  | Cấu hình: | - Image size: 512 | - Epochs: 25 | - Class: Mobile-phone | - Cùng train/validation/test split |  | Nhận xét: | - YOLO26n nhẹ, nhanh, phù hợp baseline/fallback | - RT-DETR-L cho Recall và mAP tốt hơn | - Chọn RT-DETR-L để fine-tune sâu hơn
### Slide 10
- FINETUNE MODEL RTDETR-L
### Slide 11
- FINETUNE MODEL RTDETR-L
- Input Dataset | - Roboflow Mobile-phone v7 | - Format: YOLO detection | - Class: Mobile-phone | - Train/Val/Test: 4500/326/185 | - Bad labels: 0 | 2. Training Config | - Base model: RT-DETR-L pretrained | - Image size: 512 | - Epochs: 35 | - Freeze: 5 layers | - Platform: Kaggle GPU | - Goal: improve phone detection bbox | 3. Output Model | - Best model: phone_rtdetr_l.pt | - Selected epoch: 8 | - mAP50-95: 0.80077 | - Integrated into project | - Used in /phone-test and upload analysis
### Slide 12
- XÂY DỰNG HỆ THỐNG
### Slide 13
- XÂY DỰNG HỆ THỐNG
- 3. Các trang chính
### Slide 14
- XÂY DỰNG HỆ THỐNG
- 3. Các trang chính
### Slide 15
- XÂY DỰNG HỆ THỐNG
- 3. Các trang chính
### Slide 16
- XÂY DỰNG HỆ THỐNG
- 3. Các trang chính
### Slide 17
- XÂY DỰNG HỆ THỐNG
- 2. Luồng Realtime Webcam | Người dùng nhấn Start Monitoring trên trang realtime. | Browser gửi start_realtime_session qua SocketIO. | Server mở OpenCV webcam theo camera index cấu hình và chạy background loop. | Mỗi frame được xử lý bằng MediaPipe; snapshot JPEG được gửi ở tần suất thấp nếu bật. | Nếu realtime YOLO được bật, detector điện thoại chạy ở FPS thấp để bảo vệ hiệu năng. | Khi Stop Monitoring, camera được giải phóng, phiên được lưu và người dùng mở được session detail.
### Slide 18
- HẠN CHẾ HIỆN TẠI
- Drowsiness và distraction hiện dựa trên landmark/rule-based, chưa train mô hình riêng cho các hành vi này. | Detector điện thoại pretrained có thể chưa tối ưu cho góc nhìn trong xe; cần fine-tune nếu muốn kết quả mạnh hơn. | Realtime chỉ gửi snapshot JPEG tần suất thấp, không phải full video streaming. | Một số đánh giá định lượng cần ground truth thủ công để tính Precision, Recall, F1 theo sự kiện. | Báo cáo tiếng Việt có thể tiếp tục cải thiện theo hướng sinh nội dung tự nhiên hơn thay vì dịch
### Slide 19
- HƯỚNG PHÁT TRIỂN
- Fine-tune YOLO hoặc RT-DETR cho class phone trên dataset trong bối cảnh tài xế. | Bổ sung evidence snapshot cho video tải lên nếu cần minh chứng trực quan đầy đủ hơn. | Tạo bộ ground truth cho các video demo để đánh giá định lượng theo event-level. | Tối ưu realtime bằng giảm resolution, batch nhẹ, threading ổn định và kiểm soát detector FPS. | Bổ sung test tự động cho route, database, scoring và workflow upload/realtime. | Đóng gói Docker hoặc hướng dẫn triển khai offline đầy đủ cho máy demo.
- KẾT LUẬN: Dự án AI Driver Safety Assessment System không chỉ là một mô hình nhận diện đơn lẻ mà là một hệ thống phần mềm hoàn chỉnh cho bài toán giám sát an toàn tài xế. Hệ thống kết nối dữ liệu video/webcam, phân tích thị giác máy tính, theo dõi sự kiện theo thời gian, tính điểm rủi ro và trình bày kết quả bằng dashboard, timeline, báo cáo và evaluation CSV. Đây là nền tảng phù hợp để demo, bảo vệ và tiếp tục mở rộng bằng mô hình phát hiện điện thoại đã fine-tune.
### Slide 20
- (No text detected)

## Media Files
- Total media files: 34
- `ppt/media/image24.png`: 153.5 KB
- `ppt/media/image22.png`: 85.4 KB
- `ppt/media/image1.png`: 105.1 KB
- `ppt/media/image2.png`: 790.8 KB
- `ppt/media/image3.png`: 22.8 KB
- `ppt/media/image4.png`: 40.7 KB
- `ppt/media/image5.emf`: 2.5 KB
- `ppt/media/image6.png`: 48.5 KB
- `ppt/media/image7.png`: 0.6 KB
- `ppt/media/image8.jpeg`: 13.1 KB
- `ppt/media/image9.png`: 722.6 KB
- `ppt/media/image10.gif`: 1002.3 KB
- `ppt/media/image11.png`: 15.0 KB
- `ppt/media/image13.png`: 545.4 KB
- `ppt/media/image14.png`: 688.8 KB
- `ppt/media/image15.png`: 827.8 KB
- `ppt/media/image16.png`: 235.8 KB
- `ppt/media/image17.png`: 87.1 KB
- `ppt/media/image18.png`: 82.0 KB
- `ppt/media/image19.png`: 1295.4 KB
- `ppt/media/image25.png`: 137.8 KB
- `ppt/media/image26.png`: 83.5 KB
- `ppt/media/image21.png`: 779.2 KB
- `ppt/media/image12.png`: 14.4 KB
- `ppt/media/image20.png`: 98.6 KB
- `ppt/media/image27.png`: 36.4 KB
- `ppt/media/image28.png`: 156.9 KB
- `ppt/media/image23.png`: 313.2 KB
- `ppt/media/image29.jpeg`: 172.2 KB
- `ppt/media/image30.jpeg`: 113.8 KB
- `ppt/media/image31.jpeg`: 31.5 KB
- `ppt/media/image32.png`: 20.7 KB
- `ppt/media/image33.png`: 11.5 KB
- `ppt/media/image34.svg`: 7.5 KB