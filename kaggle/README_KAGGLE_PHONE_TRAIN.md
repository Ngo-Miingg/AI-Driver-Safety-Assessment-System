# Kaggle Phone Detection Training

## Files

- Notebook: `kaggle/phone_detection_yolo11_kaggle_train.ipynb`
- Dataset zip for Kaggle: `dataset/kaggle_upload/mobilephone_wusj2_v7_yolo11_kaggle.zip`

## Upload to Kaggle

1. Open Kaggle.
2. Create a new Dataset.
3. Upload `mobilephone_wusj2_v7_yolo11_kaggle.zip`.
4. Create a new Notebook.
5. Add the uploaded dataset to the Notebook input.
6. Import `phone_detection_yolo11_kaggle_train.ipynb`.
7. Set accelerator to GPU T4 x2.
8. Turn Internet on for the first run if `ultralytics` is not preinstalled.
9. Run all cells.

## Output

The notebook writes:

- `/kaggle/working/phone_detection_export/phone_yolo11_best.pt`
- `/kaggle/working/phone_detection_export/phone_yolo11_last.pt`
- `/kaggle/working/phone_detection_training_results.zip`

Download `phone_detection_training_results.zip`, then copy `phone_yolo11_best.pt` into:

```text
D:\Work\HocKy3\ComputerVision\MDDAD\models\phone_yolo11n.pt
```

Then test it in the app:

```text
http://127.0.0.1:5000/phone-test
```

Choose `YOLO fine-tuned`, or paste the model path in `Custom model path`.

## Notes

- The dataset is one-class YOLO: `0 = Mobile-phone`.
- The Kaggle-ready `data.yaml` uses relative paths that work after Kaggle dataset extraction.
- If T4 x2 runs out of memory, reduce `BATCH` from `32` to `16`.
- If training is too slow, reduce `EPOCHS` from `80` to `50`.
