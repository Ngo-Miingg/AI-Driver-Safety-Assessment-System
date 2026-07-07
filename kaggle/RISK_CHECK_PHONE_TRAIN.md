# Risk Check - Phone Detection Kaggle Training

## Fixed before upload

1. Kaggle dataset path risk
   - Risk: Kaggle may keep the uploaded dataset as a `.zip` instead of extracting it.
   - Fix: Notebook now searches for `data.yaml`; if missing, it automatically extracts the first `.zip` found under `/kaggle/input`.

2. Wrong `data.yaml` paths
   - Risk: Roboflow export used paths such as `../train/images`, which can fail after Kaggle extraction.
   - Fix: Kaggle dataset zip contains a corrected `data.yaml`:
     - `train: train/images`
     - `val: valid/images`
     - `test: test/images`

3. T4 x2 multi-GPU instability
   - Risk: Ultralytics DDP with `device='0,1'` can fail in some Kaggle sessions.
   - Fix: Notebook tries T4 x2 first, then falls back to single GPU with smaller batch if multi-GPU training fails.

4. App class-name mismatch
   - Risk: Trained model class is `Mobile-phone`; app previously recognized `cell phone`, `mobile phone`, `phone`, `cellphone`.
   - Fix: App detector now also recognizes `mobile-phone` and `mobilephone` as phone classes.

5. Dataset integrity
   - Checked:
     - Train: 9009 images / 9009 labels
     - Valid: 326 images / 326 labels
     - Test: 185 images / 185 labels
     - Bad label lines: 0
     - Missing labels: 0

## Risks still to watch during training

1. Internet dependency
   - YOLO11 pretrained weights and `ultralytics` may need internet on first Kaggle run.
   - Mitigation: Turn Kaggle Internet on. If offline, add the pretrained weight file as another Kaggle dataset.

2. GPU memory
   - Default config uses `batch=32`, `imgsz=640`.
   - If out-of-memory occurs, reduce `BATCH` to `16` or `8`.

3. Domain gap
   - Dataset contains general mobile-phone images, not only in-car driver views.
   - Mitigation: After baseline training, test on driver/cabin videos and fine-tune with driver-phone data if needed.

4. False positives
   - The model may detect phones outside the driver region.
   - Mitigation: Use confidence threshold and later add driver ROI / face-phone spatial logic.

5. Output integration
   - Copy `phone_yolo11_best.pt` to:
     `D:\Work\HocKy3\ComputerVision\MDDAD\models\phone_yolo11n.pt`
   - Then test with `/phone-test` before enabling it in upload or realtime workflows.

