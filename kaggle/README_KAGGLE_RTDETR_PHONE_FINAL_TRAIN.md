# Final RT-DETR Phone Detection Training

Use this notebook to train the main model for the project.

## Input dataset

Use the full dataset zip:

`dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_kaggle.zip`

## Training profile

- Model: `rtdetr-l.pt`
- Dataset: full Roboflow v7 split
- Train images: 9009
- Validation images: 326
- Test images: 185
- Image size: 512
- Epochs: 25
- Patience: 8
- Device: single GPU for stability
- Batch retry: 4, 2, 1

## Output

Download `/kaggle/working/phone_rtdetr_final_training_results.zip`.

Copy `phone_rtdetr_l_best.pt` to:

`models/phone_rtdetr_l.pt`
