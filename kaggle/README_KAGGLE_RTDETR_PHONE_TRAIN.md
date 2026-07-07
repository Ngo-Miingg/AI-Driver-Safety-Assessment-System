# Kaggle RT-DETR Phone Training

Use this package to fine-tune RT-DETR-L for phone detection.

## Files

- `phone_detection_rtdetr_kaggle_train.ipynb`: Kaggle notebook.
- `dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_kaggle.zip`: dataset upload file.

## Kaggle steps

1. Upload `mobilephone_wusj2_v7_rtdetr_kaggle.zip` as a Kaggle Dataset.
2. Create a Kaggle Notebook and import `phone_detection_rtdetr_kaggle_train.ipynb`.
3. Add the uploaded dataset to the notebook input.
4. Set Accelerator to `GPU T4 x2` and enable Internet for first install.
5. Run all cells.
6. Download `/kaggle/working/phone_rtdetr_training_results.zip`.
7. Copy `phone_rtdetr_l_best.pt` to `models/phone_rtdetr_l.pt` in the local project.

## Recommended model

`rtdetr-l.pt` is used because it balances real-time inference and detection quality better than `rtdetr-x.pt` for this project.

