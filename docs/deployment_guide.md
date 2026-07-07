# Deployment Guide

## Local Defense Deployment

1. Create a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and adjust paths if needed.
4. Place optional detector models in `models/`.
5. Run:

```powershell
python app.py
```

6. Open `http://127.0.0.1:5000`.

## Optional Models

No model is mandatory. To enable advanced phone detection later:

- YOLO fine-tuned: `models/phone_yolo11n.pt`
- RT-DETR fine-tuned: `models/phone_rtdetr_r18.pt`

Set:

```text
PHONE_DETECTOR_BACKEND=yolo_finetuned
```

or:

```text
PHONE_DETECTOR_BACKEND=rtdetr_finetuned
```

## Production Notes

For real deployment, replace the dev server, use a production database, rotate secrets, disable seeded demo user, and add HTTPS/session hardening.
