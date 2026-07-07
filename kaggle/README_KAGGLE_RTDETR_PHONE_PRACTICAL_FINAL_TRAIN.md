# Practical Final RT-DETR Phone Fine-Tuning

This is the recommended notebook when the full RT-DETR-L run is too slow on Kaggle T4.

## Files

- Notebook: `phone_detection_rtdetr_practical_final_kaggle_train.ipynb`
- Dataset: `dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_practical_final_kaggle.zip`

## Training profile

- Model: `rtdetr-l.pt`
- Train images: 3000 representative images
- Validation images: 326
- Test images: 185
- Image size: 512
- Epochs: 8
- Batch retry: 16, 12, 8, 4
- Freeze layers: 10
- Output: `phone_rtdetr_l_best.pt`

This is a practical fine-tuning run, not a from-scratch full training run.
