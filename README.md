# Driver Safety Monitoring System

Ung dung Flask chay cuc bo de quan ly tai xe, giam sat webcam va ghi nhan
su kien mat tap trung, buon ngu va su dung dien thoai. Sau giai doan thu
nghiem nhieu model, du an hien duoc giu theo huong **san pham runtime gon**:

- `driver_safety_v1`: pipeline chinh cho khuon mat, dac trung thoi gian va
  phat hien buon ngu.
- `rtdetr_finetuned`: detector dien thoai doc lap dung de canh bao som va luu
  evidence bbox.
- `driver-action-recognition-adas-0002`: model hanh vi theo clip, dung nhu
  nhanh bo sung de xac nhan ngu canh hanh vi.

Nhung bundle nghien cuu cu nhu MMRT, Activity7, VITDD va cac route test phuc
tap khong con la mot phan cua runtime mac dinh. Chung chi duoc giu lai tren dia
de doi chieu, bao cao hoac tai lap nghien cuu khi can.

## Runtime hien tai

Luot xu ly chinh cua realtime:

```text
Camera -> Face detector -> Face landmarks/features -> Temporal drowsiness model
       -> 16-frame OpenVINO action model -----------+-> Event manager
       -> RT-DETR phone presence (early warning) ---+-> Risk + UI
                                                    `-> SQLite/history/report
```

Nhung gi dang hoat dong trong app chinh:

- Dang nhap, quan ly va chon tai xe.
- Webcam realtime qua OpenCV + Flask-SocketIO.
- YOLO face detector.
- MediaPipe Face Mesh trich xuat 6 dac trung:
  `left_ear`, `right_ear`, `mar`, `pitch`, `yaw`, `roll`.
- `StandardScaler` goc + `best_model.h5` nhan chuoi `15 x 6` de suy luan buon ngu.
- OpenVINO `driver-action-recognition-adas-0002` cho 9 lop hanh vi theo clip;
  day la nguon xac nhan su kien hanh vi.
- RT-DETR-L fine-tuned cho canh bao dien thoai som va bbox; detector nay khong
  tu tao su kien da xac nhan.
- Mot temporal event manager duy nhat xac nhan tin hieu lien tuc truoc khi ghi
  event vao SQLite.
- Session, event, risk log, history va report luu vao SQLite.

Nhung gi khong con duoc xem la product runtime:

- MMRT full v2 t4x2.
- Activity7 multimodal package.
- VITDD behavior branch.
- MediaPipe/OpenCV rule-only demo cu.
- YOLO pretrained va cac trang test nghien cuu cu.

## Cau truc chinh

```text
MDDAD/
|-- app.py
|-- config.py
|-- mddad/
|   |-- realtime_monitor.py
|   |-- realtime_event_manager.py
|   |-- driver_safety_v1.py
|   |-- driver_action_openvino.py
|   |-- face_identity.py
|   `-- ...
|-- templates/
|-- static/
|-- models/
|   |-- driver_safety_v1/
|   |-- openvino_driver_action_adas_0002/
|   `-- phone_rtdetr_l.pt
|-- uploads/
|-- instance/
`-- tests/
```

## Cai dat

Khuyen nghi Python 3.11:

```powershell
cd D:\Work\HocKy3\ComputerVision\MDDAD
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Neu `Activate.ps1` khong ton tai hoac PowerShell chan script, van co the chay
truc tiep bang `.\.venv\Scripts\python.exe`.

## Chay ung dung

```powershell
cd D:\Work\HocKy3\ComputerVision\MDDAD
.\.venv\Scripts\python.exe app.py
```

Mo:

```text
http://127.0.0.1:5000
```

Tai khoan demo:

```text
admin / admin123
```

## Kiem tra truoc khi demo

```powershell
cd D:\Work\HocKy3\ComputerVision\MDDAD
.\.venv\Scripts\python.exe -m compileall -q mddad config.py app.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Endpoint huu ich:

```text
/health
/debug/camera
/realtime
```

## Ghi chu cleanup

Du an hien van con nhieu artifact nghien cuu, script tao slide, notebook Kaggle,
report va file media. Truoc khi ban giao cho nguoi khac, can chia ro:

- source runtime
- model bundle runtime
- dataset/train artifact
- tai lieu bao cao

Checklist hien tai nam o:

- `docs/PROJECT_CLEANUP_AUDIT.md`
- `docs/PRODUCT_ROADMAP.md`
- `docs/CLEANUP_FOR_HANDOFF.md`
