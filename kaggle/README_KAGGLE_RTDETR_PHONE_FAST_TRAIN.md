# Fast Kaggle RT-DETR Phone Training

This is the recommended quick training package for a 1-2 hour Kaggle demo run.

## Files

- `phone_detection_rtdetr_fast_kaggle_train.ipynb`
- `dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_fast_kaggle.zip`

## Training profile

- Model: `rtdetr-l.pt`
- Train subset: 2000 images
- Validation: 326 images
- Test: 185 images
- Image size: 416
- Epochs: 25
- Batch retry: 4, 2, 1

This is faster than full training and is suitable for demo validation. For final accuracy, run the full dataset notebook later.
