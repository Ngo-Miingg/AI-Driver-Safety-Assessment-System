# Script thuyet trinh ngan gon

## Slide 1 - Gioi thieu

Kinh chao quy thay co va ban giam khao. Nhom em xin trinh bay de tai **AI Driver Safety Assessment System**, he thong danh gia an toan tai xe bang thi giac may tinh. Muc tieu cua nhom la xay dung mot san pham co the demo duoc tu dau vao video hoac webcam den dau ra la diem an toan, canh bao va bao cao phien lai.

## Slide 2 - Dat van de

Trong thuc te, nhieu tai nan giao thong xay ra do tai xe buon ngu, mat tap trung hoac su dung dien thoai. Cac hanh vi nay co the chi xuat hien trong vai giay, nhung neu lap lai nhieu lan se tao ra rui ro lon. Vi vay, nhom em de xuat mot he thong dung camera de theo doi cac dau hieu nguy hiem va danh gia muc do an toan cua tai xe theo thoi gian.

## Slide 3 - Muc tieu

Muc tieu tong quat cua du an la xay dung mot he thong phan mem hoan chinh. He thong nhan dau vao tu video tai len hoac webcam realtime, sau do phan tich khuon mat, mat, mieng, huong dau va dien thoai. Ket qua duoc chuyen thanh timeline su kien, diem rui ro, diem an toan, cap canh bao va bao cao giai thich.

## Slide 4 - Thong tin trien khai

Ve mat trien khai, he thong duoc xay dung theo huong web application. Backend su dung Flask, du lieu luu bang SQLite, xu ly video bang OpenCV, phan tich landmark bang MediaPipe va giao tiep realtime bang Flask-SocketIO. Cach lam nay giup he thong vua co the phan tich video tai len, vua co the giam sat webcam truc tiep.

## Slide 5 - Lo trinh trien khai

Nhom em trien khai theo tung giai doan. Dau tien la khao sat bai toan va chuan bi du lieu. Sau do xay dung cac thanh phan phan tich buon ngu, mat tap trung va dien thoai. Tiep theo la phan web dashboard, realtime webcam, lich su, bao cao va danh gia. Cuoi cung la hoan thien demo, slide va tai lieu bao ve.

## Slide 6 - Dataset

Du an hien su dung ba nhom du lieu. Thu nhat la dataset phat hien dien thoai theo dinh dang YOLO voi mot class la **cell phone**, dung de fine-tune detector. Thu hai la nhom dataset hanh vi tai xe nhu Drive&Act hoac DMD, phu hop cho huong phat trien multimodal action recognition. Thu ba la cac video demo de kiem thu cac tinh huong nhu binh thuong, buon ngu, ngap, nhin lech, cui dau va dung dien thoai.

## Slide 7 - Lua chon phuong phap

He thong khong chi dung mot model don le ma ket hop nhieu nhanh xu ly. MediaPipe Face Mesh duoc dung de tinh EAR, MAR va Head Pose, tu do phat hien nham mat, ngap, nhin lech va cui dau. Phone Detection dung de phat hien dien thoai. Temporal Event Tracking theo doi hanh vi theo thoi gian de giam canh bao gia. Cuoi cung, Risk Engine tong hop cac su kien de tao Risk Score, Safety Score va Alert Level.

## Slide 8 - Xay dung he thong: kien truc tong quat

Luong xu ly cua he thong bat dau tu nguoi dung dang nhap va chon tai xe. Sau do nguon dau vao co the la video upload hoac webcam realtime. OpenCV doc du lieu va tach frame. Cac nhanh phan tich se trich xuat dac trung khuon mat, hanh vi va dien thoai. Sau do Temporal Logic xac nhan su kien, Risk Engine tinh diem, va ket qua duoc luu vao SQLite de hien thi tren dashboard, history, report va evaluation.

## Slide 9 - Xay dung he thong: video upload

Voi luong video upload, nguoi dung chon tai xe va tai video len. He thong kiem tra file, trich xuat metadata nhu thoi luong, FPS va so frame bang OpenCV. Sau do video duoc lay mau theo sample FPS de tranh xu ly qua nang. MediaPipe va detector dien thoai se phan tich cac frame, roi ket qua duoc tong hop thanh su kien, diem rui ro va bao cao phien.

## Slide 10 - Cac trang chinh: dashboard

Trang dashboard la noi tong hop nhanh tinh hinh he thong. Nguoi dung co the xem tong so phien, diem an toan trung binh, diem gan nhat, so luong su kien buon ngu, mat tap trung va su dung dien thoai. Dashboard cung co cac lien ket nhanh den realtime, upload, history, reports, drivers va settings.

## Slide 11 - Cac trang chinh: realtime

Trang realtime dung de giam sat truc tiep bang webcam. Khi nhan Start Monitoring, browser gui su kien qua SocketIO den server. Server mo camera bang OpenCV va xu ly tung frame bang MediaPipe. Trang web cap nhat elapsed time, risk score, safety score, alert level, trang thai face detected, eye closed, yawn, looking away, head down va snapshot JPEG tan suat thap.

## Slide 12 - Cac trang chinh: session detail va report

Sau moi phien, nguoi dung co the mo trang session detail. Trang nay hien thi tom tat phien, diem an toan, diem rui ro, cap canh bao, timeline su kien, evidence snapshot neu co va bao cao giai thich. Muc tieu la giup nguoi xem khong chi thay ket qua cuoi cung, ma con hieu tai sao he thong danh gia nhu vay.

## Slide 13 - Cac trang chinh: settings va evaluation

He thong co trang settings de cau hinh cac nguong nhu EAR, MAR, thoi gian toi thieu cho mat nham, ngap, nhin lech, cui dau, mat khuon mat va FPS xu ly. Trang evaluation dung de xem cac chi so phien nhu processed frames, face detection rate, event count va export CSV. Phan nay giup he thong co kha nang hieu chinh va danh gia, thay vi chi demo cam tinh.

## Slide 14 - Luong Realtime Webcam

Luong realtime hoat dong nhu sau: nguoi dung nhan Start Monitoring, browser gui `start_realtime_session` qua SocketIO. Server mo webcam theo camera index cau hinh va chay background loop. Moi frame duoc xu ly bang MediaPipe, dong thoi snapshot JPEG duoc gui o tan suat thap de tranh bien thanh full video streaming. Khi nhan Stop Monitoring, camera duoc giai phong, phien duoc luu va nguoi dung co the mo session detail.

## Slide 15 - Han che hien tai

Hien tai he thong van con mot so han che. Cac hanh vi buon ngu va mat tap trung dang dua tren landmark va rule-based threshold, chua train model rieng. Detector dien thoai can fine-tune them tren bo du lieu trong xe de manh hon. Realtime chi gui snapshot tan suat thap, khong stream video day du. Ngoai ra, de danh gia Precision, Recall va F1 theo su kien, can bo ground truth duoc gan nhan thu cong.

## Slide 16 - Huong phat trien va ket luan

Huong phat trien tiep theo la fine-tune YOLO hoac RT-DETR cho class phone trong boi canh tai xe, bo sung ground truth cho video demo, toi uu realtime va dong goi he thong de demo offline. Ve lau dai, nhom em muon mo rong sang multimodal action recognition, ket hop video, landmark, head pose, phone detector va temporal features. Ket luan lai, day khong chi la mot model nhan dien don le, ma la mot he thong danh gia an toan tai xe hoan chinh tu dau vao den bao cao dau ra.

## Slide 17 - Ket thuc

Phan trinh bay cua nhom em den day la ket thuc. Nhom em xin cam on quy thay co va ban giam khao da lang nghe, va san sang tra loi cau hoi.

