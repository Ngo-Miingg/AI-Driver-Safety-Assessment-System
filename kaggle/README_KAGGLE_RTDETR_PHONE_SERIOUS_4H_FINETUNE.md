# Serious 4h RT-DETR Phone Fine-Tuning

This notebook is stronger than the ultralight RT-DETR run and is intended as a serious model candidate.

## Files

- Notebook: `phone_detection_rtdetr_serious_4h_finetune_kaggle.ipynb`
- Dataset: `dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_serious_4h_kaggle.zip`

## Profile

- Model: `rtdetr-l.pt`
- Train images: 4500
- Valid images: 326
- Test images: 185
- Image size: 512
- Epochs: 35
- Freeze layers: 5
- Batch retry: 16, 12, 8, 4
- Patience: 8

Use `phone_rtdetr_l_best.pt` as `models/phone_rtdetr_l.pt` in the local Flask app.
