from __future__ import annotations


BEHAVIOR_MODEL_OPTIONS = [
    {
        "backend": "driver_safety_v1",
        "label_key": "realtime.behavior_driver_safety_label",
        "label": "Driver Safety v1: face and temporal drowsiness",
        "description": (
            "YOLO face localization and source-compatible temporal BiLSTM drowsiness inference."
        ),
        "status": "available",
    }
]


PHONE_DETECTOR_OPTIONS = [
    {
        "backend": "disabled",
        "label_key": "realtime.phone_disabled_label",
        "label": "Disabled",
        "description": "Turn off phone detection for this realtime session.",
        "config_name": None,
        "trained": False,
    },
    {
        "backend": "rtdetr_finetuned",
        "label_key": "realtime.phone_rtdetr_serious_label",
        "label": "RT-DETR-L serious fine-tuned",
        "description": "Early phone-presence warning and bounding-box evidence.",
        "config_name": "RTDETR_MODEL_PATH",
        "trained": True,
    },
]


def behavior_model_options():
    return [dict(option, value=option["backend"]) for option in BEHAVIOR_MODEL_OPTIONS]


def phone_detector_options(app, include_disabled=True):
    options = []
    for option in PHONE_DETECTOR_OPTIONS:
        if not include_disabled and option["backend"] == "disabled":
            continue
        model_path = None
        config_name = option.get("config_name")
        if config_name:
            model_path = app.config.get(config_name)
        options.append(
            {
                **option,
                "value": option["backend"],
                "model_path": model_path,
            }
        )
    return options


def supported_phone_backends(include_disabled=True):
    backends = {option["backend"] for option in PHONE_DETECTOR_OPTIONS}
    if not include_disabled:
        backends.discard("disabled")
    return backends


def supported_behavior_backends():
    backends = {option["backend"] for option in BEHAVIOR_MODEL_OPTIONS}
    return backends
