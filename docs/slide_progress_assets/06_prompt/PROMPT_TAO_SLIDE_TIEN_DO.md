# Prompt Tao Slide Tien Do Du An

Ban la mot chuyen gia thiet ke slide bao ve do an AI/Computer Vision. Hay tao mot bo slide tien do hien tai cho du an **AI Driver Safety Assessment System** bang tieng Viet, phong cach hoc thuat, ro rang, hien dai, nen trang, mau chu dao xanh duong/cam theo phong cach Dai Nam University.

## Muc Tieu Slide

Tao slide the hien tien do du an hien tai, nhan manh rang he thong da co:

- Web app Flask co dang nhap, quan ly tai xe, dashboard, history, reports, settings, evaluation.
- Upload video analysis.
- Realtime webcam monitoring bang OpenCV, MediaPipe, Flask-SocketIO.
- Low-rate snapshot preview va event evidence snapshots.
- Rule-based risk score, safety score, alert level, fatigue trend, event timeline.
- Phone detector da fine-tune va da tich hop vao project.
- Da so sanh YOLO26n va RT-DETR-L tren dataset Mobile-phone.

## Yeu Cau So Luong Slide

Tao 10-14 slide, ngan gon, de thuyet trinh trong 7-10 phut. Neu can rut gon xuong 8-10 slide, hay gop phan paper, diem sang tao va huong train tiep theo vao 2 slide cuoi.

## Cau Truc Slide De Xuat

### Slide 1 - Tieu De Va Trang Thai Hien Tai

Tieu de:

`AI Driver Safety Assessment System - Tien do hien tai`

Noi dung:

- Muc tieu: giam sat va danh gia an toan tai xe tu webcam realtime va video upload.
- Trang thai: da hoan thien khung he thong, da train/fine-tune phone detector, da tich hop vao app.
- Demo account: `admin / admin123`.

### Slide 2 - Kien Truc Tong Quan

Ve so do pipeline:

`User -> Login/Driver -> Webcam realtime/Video upload -> OpenCV preprocessing -> MediaPipe Face Mesh + Phone Detector -> Temporal Logic -> Risk Engine -> SQLite -> Dashboard/History/Reports/Evaluation`

Chu thich:

- MediaPipe: mat nham, ngap, nhin lech, cui dau, mat khuon mat.
- Phone detector: YOLO26n baseline va RT-DETR-L fine-tuned.
- Temporal logic: bien tin hieu frame thanh su kien.
- Risk engine: tinh Risk Score, Safety Score, Alert Level.

### Slide 3 - Dataset Va Chuan Bi Train

Noi dung:

- Dataset: Roboflow Mobile-phone v7.
- Bai toan: 1 class `Mobile-phone`.
- Dinh dang nhan: YOLO detection format.
- Tap train/valid/test da duoc chuan hoa va kiem tra.
- Bad labels: 0.

Dung bang sau:

| Split | Images | Labels |
|---|---:|---:|
| Train serious RT-DETR | 4500 | 4500 |
| Validation | 326 | 326 |
| Test | 185 | 185 |

Co the chen file bang:

`docs/slide_progress_assets/05_tables/model_metrics_summary.md`

### Slide 4 - Cac Mo Hinh Da Train/Fine-Tune

Noi dung:

- YOLO26n: model baseline nhe, toc do cao, dung de so sanh va fallback realtime.
- RT-DETR-L compare: baseline RT-DETR tren tap practical.
- RT-DETR-L serious: model chinh duoc chon, train voi 4500 anh, imgsz 512, freeze 5.

Bang thong so:

| Model | Vai tro | Train images | Image size | Best epoch |
|---|---|---:|---:|---:|
| YOLO26n baseline | Baseline nhe | 3000 | 512 | 21 |
| RT-DETR-L compare | So sanh RT-DETR | 3000 | 512 | 14 |
| RT-DETR-L serious | Model chinh | 4500 | 512 | 8 |

### Slide 5 - So Sanh Chi So Mo Hinh

Chen bieu do:

`docs/slide_progress_assets/01_model_comparison/model_metrics_comparison.png`

Chu thich danh muc:

- **Danh muc:** Model comparison.
- **Y nghia:** So sanh Precision, Recall, mAP50 va mAP50-95 giua YOLO26n baseline, RT-DETR-L compare va RT-DETR-L serious.
- **Ket luan can hien tren slide:** RT-DETR-L serious co mAP50-95 cao nhat `0.80077`, nen duoc chon lam model chinh.

So lieu can hien thi:

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO26n baseline | 0.93237 | 0.86031 | 0.92854 | 0.76828 |
| RT-DETR-L compare | 0.93969 | 0.93286 | 0.95693 | 0.77543 |
| RT-DETR-L serious | 0.93331 | 0.92239 | 0.95497 | 0.80077 |

### Slide 6 - Toc Do Train Va Kich Thuoc Model

Chen 2 bieu do:

1. `docs/slide_progress_assets/01_model_comparison/training_time_comparison.png`

Chu thich danh muc:

- **Danh muc:** Training efficiency.
- **Y nghia:** So sanh thoi gian train log duoc cua tung model.

2. `docs/slide_progress_assets/01_model_comparison/model_size_comparison.png`

Chu thich danh muc:

- **Danh muc:** Deployment cost.
- **Y nghia:** So sanh kich thuoc file model, phuc vu quyet dinh realtime/fallback.

Ket luan:

- YOLO26n nho va nhanh hon, phu hop fallback/realtime.
- RT-DETR-L nang hon nhung cho mAP50-95 tot hon.

### Slide 7 - Duong Training Cua Model Chinh

Chen bieu do:

`docs/slide_progress_assets/02_training_curves/rtdetrl_serious_training_metrics.png`

Chu thich danh muc:

- **Danh muc:** Training curve - selected model.
- **Y nghia:** Theo doi Precision, Recall, mAP50, mAP50-95 qua cac epoch cua RT-DETR-L serious.

Ket luan can hien:

- Model hoi tu tot.
- Best epoch theo mAP50-95 la epoch 8.
- mAP50-95 tot nhat: `0.80077`.

Neu can so sanh them, chen nho 2 bieu do:

- `docs/slide_progress_assets/02_training_curves/yolo26n_baseline_training_metrics.png`
- `docs/slide_progress_assets/02_training_curves/rtdetrl_compare_training_metrics.png`

### Slide 8 - Confusion Matrix Va Anh Du Doan Mau

Chen confusion matrix:

`docs/slide_progress_assets/03_confusion_matrices/rtdetrl_serious_confusion_matrix.png`

Chu thich danh muc:

- **Danh muc:** Error analysis.
- **Y nghia:** Kiem tra loi du doan cua model chinh tren validation/test.

Chen anh predict mau, neu can:

- `docs/slide_progress_assets/04_prediction_samples/rtdetrl_compare/`
- `docs/slide_progress_assets/04_prediction_samples/yolo26n_baseline/`

Chu thich danh muc:

- **Danh muc:** Qualitative prediction examples.
- **Y nghia:** Minh hoa bounding box dien thoai tren anh test.

Neu slide qua day, chi chon 2-4 anh dep nhat.

### Slide 9 - Tich Hop Vao He Thong

Noi dung:

Model da duoc dua vao project:

- `models/phone_rtdetr_l.pt` - model chinh RT-DETR-L.
- `models/phone_yolo26n.pt` - baseline/fallback YOLO26n.
- `models/phone_yolo11n.pt` - compatibility copy.
- `models/phone_rtdetr_l_light_compare.pt` - checkpoint so sanh.

Config da cap nhat:

- `RTDETR_MODEL_PATH = models/phone_rtdetr_l.pt`
- `YOLO_FINETUNED_MODEL_PATH = models/phone_yolo26n.pt`
- `ultralytics >= 8.4.68`

Test trong app:

- Mo `/phone-test`.
- Chon `RT-DETR fine-tuned`.
- Upload anh/video co dien thoai.
- So sanh voi `YOLO fine-tuned`.

### Slide 10 - Ket Qua Hien Tai Va Viec Tiep Theo

Noi dung:

Da hoan thanh:

- Khung he thong web.
- Realtime webcam monitoring.
- Upload video analysis.
- Dashboard/history/report/evaluation.
- Fine-tune va so sanh phone detector.
- Tich hop model vao project.

Dang can test tiep:

- Test model tren video cabin that.
- So sanh toc do RT-DETR va YOLO tren `/phone-test`.
- Chon backend realtime: RT-DETR low FPS hoac YOLO26n fallback.
- Ghi nhan ket qua demo vao evaluation/report.

Ket luan:

`RT-DETR-L serious fine-tune duoc chon lam model chinh nho mAP50-95 cao nhat; YOLO26n duoc giu lam baseline va fallback toc do cao.`

### Slide 11 - Nghien Cuu Lien Quan

Noi dung:

- Nhom MediaPipe/OpenCV drowsiness detection: dung Face Mesh, EAR, MAR, head pose de phat hien mat nham, ngap, cui dau.
- Nhom distracted driver action recognition: AI City Track 3, M2DAR, multi-view action recognition, dung video clip va temporal model.
- Nhom object detection realtime: YOLO va RT-DETR cho phat hien doi tuong nhu dien thoai trong frame.

Bang tom tat nen co:

| Huong nghien cuu | Cach lam | Diem manh | Han che | Lien he du an |
|---|---|---|---|---|
| MediaPipe drowsiness | EAR/MAR/head pose | Nhanh, giai thich duoc | Phu thuoc nguong/goc camera | Dang dung cho fatigue/distraction baseline |
| Action recognition | Video clip + temporal model | Dung ban chat hanh vi | Can dataset video gan nhan | Huong train tiep theo |
| RT-DETR/YOLO detection | Object detector | Co bbox/confidence | Can can bang toc do/chinh xac | Dang dung cho phone detection |

### Slide 12 - Diem Noi Bat Va Sang Tao Cua Du An

Noi dung:

- Khong chi train model rieng le, ma da tich hop vao mot he thong DMS co workflow dau-cuoi.
- Ket hop MediaPipe/rule-based cho hanh vi khuon mat voi phone detector da fine-tune.
- Co Temporal Logic de bien tin hieu theo frame thanh su kien co thoi gian bat dau/ket thuc.
- Co Risk Engine tinh Risk Score, Safety Score va Alert Level.
- Co evidence snapshot, dashboard, history, report va evaluation CSV de phuc vu demo/defense.
- Co so sanh thuc nghiem YOLO26n va RT-DETR-L truoc khi chon model.

Ket luan can hien:

`Diem sang tao cua du an nam o viec bien cac thanh phan Computer Vision thanh mot he thong danh gia an toan tai xe co the demo, giai thich va mo rong, thay vi chi dung o muc notebook/model don le.`

### Slide 13 - Huong Train AI Tiep Theo Cho Hanh Vi Bat Thuong

Noi dung:

- Hien tai mat nham, ngap, nhin lech, cui dau dang dung MediaPipe/OpenCV + nguong cau hinh.
- Buoc tiep theo la xay dung dataset video/clip cho cac hanh vi bat thuong.
- Moi clip nen co label: normal, eye closure, yawning, looking away, head down, phone usage, mixed-risk.
- Co the train action recognition/video model: CNN-LSTM, SlowFast, Video Transformer hoac multimodal fusion.
- Output model moi se dua vao Temporal Logic va Risk Engine san co.

Thong diep can noi:

`Du an da train phone detector thanh cong; giai doan tiep theo se train cac hanh vi bat thuong de thay the/tang cuong baseline MediaPipe/OpenCV hien tai.`

### Slide 14 - Tai Lieu Tham Khao Chinh

Chi nen hien 4-5 nguon tieu bieu:

- Zhao et al., `DETRs Beat YOLOs on Real-time Object Detection`, CVPR 2024.
- AI City Challenge 2023 Track 3: Naturalistic Driving Action Recognition.
- Zhou et al., `Multi View Action Recognition for Distracted Driver Behavior Localization`, CVPRW 2023.
- PurdueDigitalTwin/M2DAR: source cho driver action recognition in untrimmed videos.
- `Development of a Real-time Driver's Drowsiness Detection System Using MediaPipe Face Mesh`.

Ghi chu:

Khong can dua qua nhieu reference len slide chinh. Neu slide qua day, dua reference vao phu luc.

## Danh Sach Assets Va Chu Thich

## Neu AI Khong Cho Upload Nhieu Anh

Hay su dung cac anh tong hop trong thu muc:

`docs/slide_progress_assets/00_contact_sheets/`

Danh sach anh tong hop:

- `01_model_comparison_sheet.png`
  - **Danh muc:** Model comparison + training efficiency + deployment cost.
  - **Noi dung:** Gom 3 bieu do: model metrics comparison, training time comparison, model size comparison.
  - **Dung cho slide:** Slide 5 va Slide 6.

- `02_training_curves_sheet.png`
  - **Danh muc:** Training curves.
  - **Noi dung:** Gom curve cua YOLO26n baseline, RT-DETR-L compare, RT-DETR-L serious.
  - **Dung cho slide:** Slide 7.

- `03_confusion_matrices_sheet.png`
  - **Danh muc:** Error analysis.
  - **Noi dung:** Gom confusion matrix cua YOLO26n, RT-DETR-L compare, RT-DETR-L serious.
  - **Dung cho slide:** Slide 8.

- `04_yolo26n_prediction_samples_sheet.png`
  - **Danh muc:** Qualitative prediction examples - YOLO baseline.
  - **Noi dung:** Anh du doan mau cua YOLO26n.
  - **Dung cho slide:** Slide 8 hoac phu luc so sanh.

- `05_rtdetrl_prediction_samples_sheet.png`
  - **Danh muc:** Qualitative prediction examples - RT-DETR.
  - **Noi dung:** Anh du doan mau cua RT-DETR-L.
  - **Dung cho slide:** Slide 8 hoac phu luc so sanh.

- `06_model_metrics_table_sheet.png`
  - **Danh muc:** Model metrics table.
  - **Noi dung:** Bang tong hop chi so Precision, Recall, mAP50, mAP50-95, time, size.
  - **Dung cho slide:** Slide 4 hoac Slide 5.

Neu chi duoc upload moi lan 1 anh, hay upload theo thu tu:

1. `01_model_comparison_sheet.png`
2. `02_training_curves_sheet.png`
3. `03_confusion_matrices_sheet.png`
4. `06_model_metrics_table_sheet.png`

Sau do neu con can minh hoa, moi upload prediction sample sheets.

### Bang so lieu

- `docs/slide_progress_assets/05_tables/model_metrics_summary.csv`
  - **Danh muc:** Model metrics table.
  - **Dung cho:** Bang so sanh chi so chi tiet.

- `docs/slide_progress_assets/05_tables/model_metrics_summary.md`
  - **Danh muc:** Model metrics table.
  - **Dung cho:** Copy nhanh vao slide/doc.

### Bieu do so sanh

- `docs/slide_progress_assets/01_model_comparison/model_metrics_comparison.png`
  - **Danh muc:** Model comparison.
  - **Dung cho:** Slide so sanh Precision, Recall, mAP50, mAP50-95.

- `docs/slide_progress_assets/01_model_comparison/training_time_comparison.png`
  - **Danh muc:** Training efficiency.
  - **Dung cho:** Slide so sanh thoi gian train.

- `docs/slide_progress_assets/01_model_comparison/model_size_comparison.png`
  - **Danh muc:** Deployment cost.
  - **Dung cho:** Slide so sanh do nang nhe khi trien khai.

### Training curves

- `docs/slide_progress_assets/02_training_curves/yolo26n_baseline_training_metrics.png`
  - **Danh muc:** Training curve - YOLO baseline.
  - **Dung cho:** Minh hoa YOLO26n hoi tu.

- `docs/slide_progress_assets/02_training_curves/rtdetrl_compare_training_metrics.png`
  - **Danh muc:** Training curve - RT-DETR comparison.
  - **Dung cho:** Minh hoa RT-DETR-L compare.

- `docs/slide_progress_assets/02_training_curves/rtdetrl_serious_training_metrics.png`
  - **Danh muc:** Training curve - selected model.
  - **Dung cho:** Minh hoa model chinh RT-DETR-L serious.

### Confusion matrices

- `docs/slide_progress_assets/03_confusion_matrices/yolo26n_baseline_confusion_matrix.png`
  - **Danh muc:** Error analysis - YOLO baseline.

- `docs/slide_progress_assets/03_confusion_matrices/rtdetrl_compare_confusion_matrix.png`
  - **Danh muc:** Error analysis - RT-DETR compare.

- `docs/slide_progress_assets/03_confusion_matrices/rtdetrl_serious_confusion_matrix.png`
  - **Danh muc:** Error analysis - selected model.

### Prediction samples

- `docs/slide_progress_assets/04_prediction_samples/yolo26n_baseline/`
  - **Danh muc:** Qualitative prediction examples - YOLO baseline.

- `docs/slide_progress_assets/04_prediction_samples/rtdetrl_compare/`
  - **Danh muc:** Qualitative prediction examples - RT-DETR compare.

## Phong Cach Thiet Ke

- Nen trang, khong nen nen toi.
- Tieu de mau cam hoac xanh dam.
- Dung icon don gian: camera, dashboard, database, detector, report.
- Moi slide khong qua 5 bullet.
- So lieu quan trong lam noi bat bang badge/card.
- Bieu do can co caption ngan: "So sanh chi so model", "Training curve model chinh", "Confusion matrix".
- Tranh dua qua nhieu chu; uu tien so do, bang, bieu do.

## Yeu Cau Dau Ra

Hay tao noi dung slide hoan chinh gom:

- Tieu de tung slide.
- Bullet noi dung.
- Vi tri chen bieu do/anh.
- Caption cho tung bieu do.
- Ghi chu nguoi thuyet trinh ngan gon ben duoi moi slide.

Khong can viet qua dai. Moi slide chi can du noi dung de thuyet trinh 30-45 giay.

---

# Prompt Bo Sung De Slide Dung Voi Du An

Su dung phan nay khi AI tao slide bi chung chung, sai model, sai luong xu ly hoac noi qua muc so voi tinh trang hien tai cua du an.

## Vai Tro Cua Ban

Ban la chuyen gia thiet ke slide tien do cho do an Computer Vision ve **AI Driver Safety Assessment System**. Hay sua/tai tao slide sao cho noi dung dung voi du an hien tai, khong them tinh nang chua co, khong phong dai thanh he thong thuong mai hoan chinh.

## Ban Chat Du An Can The Hien Dung

Day la he thong danh gia an toan tai xe dua tren thi giac may tinh, gom 2 luong dau vao:

1. **Realtime webcam monitoring**
   - Nguoi dung dang nhap, chon tai xe, bam Start Monitoring.
   - Browser ket noi server bang Flask-SocketIO.
   - Server mo webcam bang OpenCV, xu ly frame theo FPS cau hinh.
   - MediaPipe Face Mesh trich xuat landmark khuon mat.
   - He thong tinh EAR, MAR, huong mat/huong dau, mat khuon mat.
   - UI hien thi realtime metrics, alert level, timeline event va snapshot JPEG tan suat thap.
   - Khi Stop Monitoring, session duoc luu vao SQLite va co the xem chi tiet.

2. **Uploaded video analysis**
   - Nguoi dung upload video.
   - OpenCV doc metadata va tach frame theo sample FPS.
   - MediaPipe phan tich hanh vi mat nham, ngap, nhin lech, cui dau, mat khuon mat.
   - Phone detector phat hien dien thoai tren frame/video.
   - Temporal logic bien tin hieu theo frame thanh su kien da xac nhan.
   - Risk engine tinh Risk Score, Safety Score, Alert Level va bao cao.

## Diem Can Noi Ro Tren Slide

- He thong da co day du workflow web: login, driver, dashboard, upload, realtime, history, session detail, reports, settings, evaluation, CSV export.
- Drowsiness/distraction hien tai duoc phan tich bang **MediaPipe landmarks + nguong cau hinh**, khong phai model deep learning tu train.
- Phone detection la phan da train/fine-tune detector rieng.
- RT-DETR-L serious fine-tune la model chinh cho phone detection vi co mAP50-95 cao nhat.
- YOLO26n duoc giu lam baseline nhe va fallback toc do cao.
- Realtime khong gui full video stream; chi gui metrics va snapshot JPEG tan suat thap.
- Phone detection realtime nen chay FPS thap/co cau hinh de tranh lam cham webcam.
- He thong co tinh demo/defense: co bang chi so, bieu do, confusion matrix, prediction samples, report va evaluation.

## Nhung Dieu Khong Duoc Noi Sai

Khong viet cac noi dung sau:

- Khong noi he thong da train model nhan dien buon ngu/yawning/head pose rieng. Phan nay dung MediaPipe + rule/threshold.
- Khong noi realtime dang stream video full HD lien tuc. He thong chi co low-rate JPEG snapshot preview.
- Khong noi realtime RT-DETR bat buoc chay moi frame. Neu dung thi chi la tuy chon low FPS.
- Khong noi da train multimodal action model hoan chinh. Multimodal/action model chi la huong mo rong tiep theo.
- Khong noi model dat do chinh xac 100% hoac san sang trien khai thuc te quy mo lon.
- Khong noi dataset la dataset hanh vi lai xe day du. Dataset hien tai dung cho bai toan 1 class `Mobile-phone`.
- Khong thay doi ten model thanh YOLOv8/YOLOv11 neu slide dang noi ve ket qua hien tai: model so sanh chinh la `YOLO26n baseline` va `RT-DETR-L`.

## Noi Dung Chinh Can Tao Lai Cho Slide Tien Do

Hay tao lai bo slide 8-10 trang theo cau truc sau, moi slide ngan gon, uu tien so do/bang/bieu do hon chu dai:

### Slide 1 - Tien Do Hien Tai

Tieu de: **AI Driver Safety Assessment System - Tien Do Hien Tai**

Noi dung:

- Muc tieu: danh gia an toan tai xe tu webcam realtime va video upload.
- Da hoan thien khung web va workflow demo.
- Da train/fine-tune phone detector va dua vao project.
- Demo account: `admin / admin123`.

Ghi chu thuyet trinh:

> Du an khong chi dung o muc detect anh, ma da co he thong web dau-cuoi: thu thap dau vao, phan tich, cham diem rui ro, luu lich su va tao bao cao.

### Slide 2 - Kien Truc He Thong

Ve pipeline:

`User -> Login/Driver -> Webcam realtime/Video upload -> OpenCV -> MediaPipe + Phone Detector -> Temporal Logic -> Risk Engine -> SQLite -> Dashboard/History/Reports/Evaluation`

Can the hien 4 khoi quan trong:

- OpenCV: doc camera/video, tach frame.
- MediaPipe: landmark khuon mat, EAR, MAR, head pose.
- Phone detector: YOLO26n baseline va RT-DETR-L fine-tuned.
- Risk engine: tong hop su kien thanh diem rui ro/canh bao.

Ghi chu:

> Kien truc tach thanh nhieu tang de de thay model phone detection ma khong phai viet lai toan bo he thong.

### Slide 3 - Hai Luong Xu Ly Chinh

Chia slide thanh 2 cot:

Cot 1: **Realtime webcam**

- Flask-SocketIO dieu khien Start/Stop.
- OpenCV mo webcam.
- MediaPipe xu ly frame.
- UI cap nhat metrics va snapshot JPEG tan suat thap.

Cot 2: **Video upload**

- Upload video va doc metadata.
- Sample frame bang OpenCV.
- Phan tich MediaPipe va phone detector.
- Luu session, event, report, evaluation.

Ghi chu:

> Realtime uu tien toc do va do on dinh; upload video co the phan tich ky hon vi khong bi rang buoc thoi gian thuc.

### Slide 4 - Dataset Phone Detection

Noi dung:

- Dataset: Roboflow Mobile-phone v7.
- Bai toan: object detection 1 class `Mobile-phone`.
- Dinh dang: YOLO detection format.
- Dataset da duoc kiem tra: bad labels = 0.

Bang:

| Split | Images | Labels |
|---|---:|---:|
| Train serious RT-DETR | 4500 | 4500 |
| Validation | 326 | 326 |
| Test | 185 | 185 |

Ghi chu:

> Dataset nay phuc vu phan phat hien dien thoai, khong phai dataset cho toan bo hanh vi lai xe.

### Slide 5 - Mo Hinh Da Train/Fine-Tune

Noi dung:

- YOLO26n baseline: nhe, nhanh, phu hop fallback.
- RT-DETR-L compare: ban so sanh tren tap practical.
- RT-DETR-L serious: model chinh, train 4500 anh, imgsz 512.

Bang:

| Model | Vai tro | Train images | Best epoch |
|---|---|---:|---:|
| YOLO26n baseline | Baseline/fallback | 3000 | 21 |
| RT-DETR-L compare | So sanh | 3000 | 14 |
| RT-DETR-L serious | Model chinh | 4500 | 8 |

Ghi chu:

> Muc tieu khong phai chi train mot model, ma so sanh de chon model phu hop giua do chinh xac va kha nang tich hop.

### Slide 6 - Ket Qua So Sanh Model

Chen anh:

`docs/slide_progress_assets/00_contact_sheets/01_model_comparison_sheet.png`

Caption:

**So sanh Precision, Recall, mAP va chi phi trien khai cua cac model phone detection.**

So lieu can highlight:

- RT-DETR-L serious: mAP50-95 = `0.80077`, cao nhat.
- RT-DETR-L compare: Recall = `0.93286`, cao nhat.
- YOLO26n: kich thuoc `5.1 MB`, nho nhat.

Ket luan:

> RT-DETR-L serious duoc chon lam model chinh; YOLO26n duoc giu lam baseline/fallback do nhe hon nhieu.

### Slide 7 - Training Curve Va Do Hoi Tu

Chen anh:

`docs/slide_progress_assets/00_contact_sheets/02_training_curves_sheet.png`

Caption:

**Training curves cua YOLO26n, RT-DETR-L compare va RT-DETR-L serious.**

Can neu:

- RT-DETR-L serious dat best epoch o epoch 8.
- mAP50-95 tot nhat `0.80077`.
- Training khong can chay qua dai neu validation metric da dat dinh va co nguy co overfit.

Ghi chu:

> Chon best.pt dua tren validation metric, khong dua vao epoch cuoi.

### Slide 8 - Phan Tich Loi Va Du Doan Mau

Chen:

- `docs/slide_progress_assets/00_contact_sheets/03_confusion_matrices_sheet.png`
- Neu con cho: `05_rtdetrl_prediction_samples_sheet.png`

Caption:

**Confusion matrix va vi du bounding box dien thoai tren anh test.**

Noi dung:

- Kiem tra false positive/false negative.
- Minh hoa kha nang phat hien dien thoai trong anh thuc te.
- Dung prediction samples de giai thich truc quan voi hoi dong.

Ghi chu:

> Slide nay giup chung minh model khong chi co so lieu, ma co ket qua du doan co the quan sat.

### Slide 9 - Tich Hop Vao Project

Noi dung:

- Model chinh: `models/phone_rtdetr_l.pt`.
- Fallback/baseline: `models/phone_yolo26n.pt`.
- Test UI: `/phone-test`.
- Config: `RTDETR_MODEL_PATH`, `YOLO_FINETUNED_MODEL_PATH`.
- Ket qua phone detection duoc dua vao risk score, timeline, report/session detail.

Ghi chu:

> Diem manh cua du an la model da duoc dua vao ung dung co giao dien test rieng, khong chi dung o notebook.

### Slide 10 - Viec Da Hoan Thanh Va Huong Tiep Theo

Da hoan thanh:

- Web app va database.
- Realtime monitoring.
- Upload analysis.
- Evaluation/report.
- Train/fine-tune va tich hop phone detector.

Can tiep tuc:

- Test tren video cabin that.
- Do FPS RT-DETR va YOLO tren may demo.
- Chon chien luoc realtime: YOLO nhanh hoac RT-DETR low FPS.
- Neu theo huong multimodal/action model, can them dataset video/action da gan nhan.

Ket luan:

> He thong hien tai da san sang demo workflow hoan chinh; phan nang cap tiep theo la danh gia thuc nghiem tren video cabin va mo rong sang action/multimodal model neu co dataset phu hop.

## Prompt Ngan De Dung Khi Upload Tung Anh Tong Hop

Neu AI chi cho upload 1 anh moi lan, hay dung mau prompt sau:

```text
Toi dang lam slide tien do cho du an AI Driver Safety Assessment System.
Hay dung anh toi vua upload lam minh chung truc quan cho slide.

Yeu cau:
- Viet bang tieng Viet.
- Nen trang, phong cach hoc thuat, mau xanh duong/cam.
- Khong them tinh nang chua co.
- Khong noi he thong da train model buon ngu/yawning/head pose; phan do dung MediaPipe + rule/threshold.
- Khong noi realtime stream full video; realtime chi gui metrics va low-rate JPEG snapshot.
- Noi dung phai dung voi du an: Flask, SQLite, OpenCV, MediaPipe, Flask-SocketIO, phone detector YOLO26n/RT-DETR-L, risk engine, dashboard/history/report/evaluation.

Anh nay thuoc danh muc: [DIEN DANH MUC ANH]
Muc dich cua anh: [DIEN MUC DICH ANH]

Hay tao 1-2 slide tu anh nay, gom:
- Tieu de slide.
- 3-5 bullet ngan.
- Caption cho anh/bieu do.
- Ghi chu thuyet trinh 30-45 giay.
```

Goi y dien danh muc:

- Voi `01_model_comparison_sheet.png`:
  - Danh muc: **Model comparison + training efficiency + deployment cost**
  - Muc dich: **So sanh chi so, thoi gian train va kich thuoc model de giai thich vi sao chon RT-DETR-L serious lam model chinh va giu YOLO26n lam fallback.**

- Voi `02_training_curves_sheet.png`:
  - Danh muc: **Training curves**
  - Muc dich: **Minh hoa qua trinh hoi tu cua cac model va cach chon best epoch theo validation metric.**

- Voi `03_confusion_matrices_sheet.png`:
  - Danh muc: **Error analysis**
  - Muc dich: **Phan tich loi du doan va do tin cay cua model tren tap validation/test.**

- Voi `04_yolo26n_prediction_samples_sheet.png`:
  - Danh muc: **Qualitative prediction examples - YOLO baseline**
  - Muc dich: **Minh hoa YOLO26n la baseline nhe, co kha nang phat hien dien thoai va phu hop fallback toc do cao.**

- Voi `05_rtdetrl_prediction_samples_sheet.png`:
  - Danh muc: **Qualitative prediction examples - RT-DETR selected model**
  - Muc dich: **Minh hoa bounding box dien thoai cua model RT-DETR-L fine-tuned duoc chon lam model chinh.**

- Voi `06_model_metrics_table_sheet.png`:
  - Danh muc: **Model metrics table**
  - Muc dich: **Trinh bay bang so lieu tong hop de hoi dong thay ro ket qua train/fine-tune.**
