# Combined Lightweight YOLO vs RT-DETR Comparison

Use this single notebook when you want to start a run and leave it unattended.

Notebook:

`phone_detection_combined_light_comparison_kaggle.ipynb`

Dataset to upload:

`dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_practical_final_kaggle.zip`

Outputs on Kaggle:

- `/kaggle/working/compare_outputs/yolo_light/`
- `/kaggle/working/compare_outputs/rtdetr_light/`
- `/kaggle/working/phone_detector_combined_light_comparison_results.zip`

Moderate configuration:

- YOLO: `yolo26n.pt`, fallback `yolo11n.pt`, 8 epochs, imgsz 512
- RT-DETR: `rtdetr-l.pt`, 5 epochs, imgsz 512, freeze 10 layers

