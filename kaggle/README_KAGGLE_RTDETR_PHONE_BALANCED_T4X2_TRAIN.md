# RT-DETR Phone Training - Kaggle T4x2 DDP

This notebook uses both T4 GPUs with `device='0,1'`.

Use the same balanced dataset zip:

`dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_balanced_kaggle.zip`

Training profile:

- Model: `rtdetr-l.pt`
- Image size: 512
- Epochs: 15
- Global batch retry: 4, 2
- Device: `0,1`

This can use both GPUs, but it can still fail with DDP/OOM on Kaggle. If it fails, use the single-GPU balanced notebook.
