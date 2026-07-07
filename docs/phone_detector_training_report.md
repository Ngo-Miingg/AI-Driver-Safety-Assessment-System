# Phone Detector Training Report

Date: 2026-06-17

## Inputs

Training outputs were reviewed from:

- `D:\Work\output\results_finetune_rtdetr.zip`
- `D:\Work\output\results_test_yolo_rtdetr.zip`

Extracted working folder:

- `D:\Work\output_extracted`

Dataset family:

- Roboflow Mobile-phone v7
- Class count: 1
- Class name: `Mobile-phone`

## Models Reviewed

| Model | Role | Training config | Weight used |
|---|---|---|---|
| YOLO26n | Lightweight baseline | 3000 train images, 512 image size, 25 epochs | `phone_yolo_light_best.pt` |
| RT-DETR-L compare | RT-DETR baseline | 3000 train images, 512 image size, freeze 10, 17 logged epochs | `phone_rtdetr_l_best.pt` |
| RT-DETR-L serious | Main model candidate | 4500 train images, 512 image size, freeze 5, 16 logged epochs | `best.pt` |

## Validation Metrics

Best row selected by `metrics/mAP50-95(B)` from each `results.csv`.

| Model | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Training time logged |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n baseline | 21 | 0.93237 | 0.86031 | 0.92854 | 0.76828 | 665.055 s |
| RT-DETR-L compare | 14 | 0.93969 | 0.93286 | 0.95693 | 0.77543 | 1952.2 s |
| RT-DETR-L serious | 8 | 0.93331 | 0.92239 | 0.95497 | 0.80077 | 2993.76 s |

## Decision

The selected main model is **RT-DETR-L serious fine-tune**.

Reason:

- Highest mAP50-95: `0.80077`
- Strong mAP50: `0.95497`
- Good recall: `0.92239`, important for not missing phone usage
- Better localization quality than YOLO26n baseline based on mAP50-95

YOLO26n is kept as a lightweight baseline/fallback because it is much smaller and faster.

## Integrated Files

Copied into the project:

- `models/phone_rtdetr_l.pt` - main RT-DETR phone detector
- `models/phone_rtdetr_l_light_compare.pt` - RT-DETR comparison checkpoint
- `models/phone_yolo26n.pt` - YOLO26n baseline/fine-tuned detector
- `models/phone_yolo11n.pt` - compatibility copy for older config references

Config updates:

- `RTDETR_MODEL_PATH` remains `models/phone_rtdetr_l.pt`
- `YOLO_FINETUNED_MODEL_PATH` now defaults to `models/phone_yolo26n.pt`
- `requirements.txt` now requires `ultralytics>=8.4.68` to support YOLO26-era weights

## How To Test In App

1. Start the app:

```powershell
cd D:\Work\HocKy3\ComputerVision\MDDAD
python app.py
```

2. Open:

```text
http://127.0.0.1:5000/phone-test
```

3. Test RT-DETR:

- Select `RT-DETR fine-tuned`
- Upload a phone image or phone-use video
- Confirm bounding boxes appear

4. Test YOLO baseline:

- Select `YOLO fine-tuned`
- Upload the same media
- Compare confidence, boxes, and speed

## Remaining Manual Evaluation

The CSV metrics are validation metrics from training logs. Test split curves and prediction images were exported, but manual review is still needed:

- Inspect sample prediction images in `runs/*predict_samples`
- Test real driver/cabin videos in `/phone-test`
- Compare false negatives between YOLO26n and RT-DETR-L
- Pick app runtime backend based on accuracy/speed tradeoff

Recommended production/demo default:

- Uploaded Full AI: `RT-DETR fine-tuned` when accuracy matters
- Realtime: keep detector sampling low, around 1 FPS, if RT-DETR is used
- Faster realtime fallback: `YOLO fine-tuned`
