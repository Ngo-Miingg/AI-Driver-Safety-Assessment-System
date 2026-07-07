# Lightweight Model Comparison for Phone Detection

Use these notebooks before deciding the final fine-tuning direction.

## Shared dataset

Use the same practical-final dataset for both notebooks:

`dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_practical_final_kaggle.zip`

It contains 3000 train images, 326 validation images, and 185 test images.

## Notebook A: YOLO latest/lightest

`phone_detection_yolo26n_light_kaggle_train.ipynb`

- Preferred model: `yolo26n.pt`
- Fallback if unsupported by Kaggle package: `yolo11n.pt`
- Epochs: 12
- Image size: 512
- Batch retry: 64, 48, 32, 16

## Notebook B: RT-DETR lightest official Ultralytics option

`phone_detection_rtdetr_l_light_kaggle_train.ipynb`

- Model: `rtdetr-l.pt`
- Epochs: 8
- Image size: 512
- Batch retry: 16, 12, 8, 4
- Freeze layers: 10

## Compare using

- mAP50
- mAP50-95
- Precision
- Recall
- Training time
- Inference speed
- Sample prediction images

