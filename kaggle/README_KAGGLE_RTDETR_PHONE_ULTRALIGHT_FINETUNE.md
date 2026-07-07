# Ultralight RT-DETR Phone Fine-Tuning

This notebook is lighter than the practical-final RT-DETR notebook.

## Files

- Notebook: `phone_detection_rtdetr_ultralight_finetune_kaggle.ipynb`
- Dataset: `dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_ultralight_finetune_kaggle.zip`

## Profile

- Model: `rtdetr-l.pt`
- Train images: 1500
- Valid images: 326
- Test images: 185
- Image size: 416
- Epochs: 5
- Freeze layers: 20
- Batch retry: 24, 16, 12, 8, 4

This is a quick RT-DETR fine-tuning run for comparison/demo, not the strongest final model.
