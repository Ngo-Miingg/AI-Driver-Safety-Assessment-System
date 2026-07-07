# Model Metrics Summary

| Model | Role | Train images | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Time | Size MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n baseline | Lightweight baseline / realtime fallback | 3000 | 21 | 0.93237 | 0.86031 | 0.92854 | 0.76828 | 665.1s | 5.1 |
| RT-DETR-L compare | RT-DETR comparison checkpoint | 3000 | 14 | 0.93969 | 0.93286 | 0.95693 | 0.77543 | 1952.2s | 63.1 |
| RT-DETR-L serious | Selected main phone detector | 4500 | 8 | 0.93331 | 0.92239 | 0.95497 | 0.80077 | 2993.8s | 63.1 |
