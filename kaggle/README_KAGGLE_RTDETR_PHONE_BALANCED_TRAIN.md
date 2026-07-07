# Fast-Balanced Kaggle RT-DETR Phone Training

Recommended package when you want a faster run while keeping better accuracy than the 416px fast draft.

## Files

- `phone_detection_rtdetr_balanced_kaggle_train.ipynb`
- `dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_balanced_kaggle.zip`

## Training profile

- Model: `rtdetr-l.pt`
- Train subset: 1500 images
- Validation: 326 images
- Test: 185 images
- Image size: 512
- Epochs: 15
- Batch retry: 4, 2, 1
- Device: single GPU for stability

This is the recommended baseline for a quick but credible RT-DETR result.
