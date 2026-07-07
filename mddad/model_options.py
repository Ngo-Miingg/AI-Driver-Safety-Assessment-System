from __future__ import annotations


BEHAVIOR_MODEL_OPTIONS = [
    {
        "backend": "mmrt_multimodal_full_v2_t4x2",
        "label_key": "realtime.behavior_multimodal_label",
        "label": "MMRT multimodal full v2 t4x2",
        "description": (
            "Live multimodal model using the MMRT full v2 t4x2 research bundle with video, audio, face, and skeleton cues."
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
        "description": "Main RT-DETR-L phone detector trained on the serious fine-tuned dataset.",
        "config_name": "RTDETR_MODEL_PATH",
        "trained": True,
    },
    {
        "backend": "rtdetr_light_compare",
        "label_key": "realtime.phone_rtdetr_compare_label",
        "label": "RT-DETR-L comparison checkpoint",
        "description": "Comparison RT-DETR-L checkpoint kept for side-by-side evaluation.",
        "config_name": "RTDETR_LIGHT_COMPARE_MODEL_PATH",
        "trained": True,
    },
    {
        "backend": "yolo_finetuned",
        "label_key": "realtime.phone_yolo_finetuned_label",
        "label": "YOLO26n fine-tuned",
        "description": "Light phone detector fallback for lower-latency realtime monitoring.",
        "config_name": "YOLO_FINETUNED_MODEL_PATH",
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
    backends.add("activity7_multimodal_live")
    return backends
