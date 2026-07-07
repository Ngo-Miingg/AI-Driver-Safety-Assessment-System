import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-change-this-secret-key")
    DATABASE_PATH = os.environ.get(
        "DATABASE_PATH",
        str(BASE_DIR / "instance" / "mddad.sqlite3"),
    )
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 200 * 1024 * 1024))
    SOCKETIO_ASYNC_MODE = os.environ.get("SOCKETIO_ASYNC_MODE", "threading")
    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"
    USE_RELOADER = os.environ.get("FLASK_USE_RELOADER", "0") == "1"
    SEED_DEMO_USER = os.environ.get("SEED_DEMO_USER", "1") == "1"
    DEMO_USERNAME = os.environ.get("DEMO_USERNAME", "admin")
    DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "admin123")
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
    MULTIMODAL_MODEL_BACKEND = os.environ.get("MULTIMODAL_MODEL_BACKEND", "mmrt_multimodal_full_v2_t4x2")
    MULTIMODAL_MODEL_PATH = os.environ.get(
        "MULTIMODAL_MODEL_PATH",
        str(Path("D:/Downloads/mmrt_artifacts_bundle.zip")),
    )
    MMRT_V2_MODEL_PATH = os.environ.get(
        "MMRT_V2_MODEL_PATH",
        str(Path("D:/Downloads/mmrt_artifacts_bundle.zip")),
    )
    MMRT_V2_DEVICE = os.environ.get("MMRT_V2_DEVICE", "cpu")
    MMRT_V2_CLIP_SECONDS = float(os.environ.get("MMRT_V2_CLIP_SECONDS", "3.0"))
    MMRT_V2_STRIDE_SECONDS = float(os.environ.get("MMRT_V2_STRIDE_SECONDS", "1.5"))
    MMRT_V2_FRAME_COUNT = int(os.environ.get("MMRT_V2_FRAME_COUNT", "16"))
    MMRT_V2_RUN_LOG_DIR = os.environ.get(
        "MMRT_V2_RUN_LOG_DIR",
        str(BASE_DIR / "uploads" / "mmrt_v2_demo_runs"),
    )
    MULTIMODAL_DEVICE = os.environ.get("MULTIMODAL_DEVICE", os.environ.get("MMRT_V2_DEVICE", "cpu"))
    ACTIVITY7_MODEL_PACKAGE_PATH = os.environ.get(
        "ACTIVITY7_MODEL_PACKAGE_PATH",
        str(BASE_DIR / "models" / "activity7_final_best_model_package.zip"),
    )
    ACTIVITY7_DECISION_MODE = os.environ.get("ACTIVITY7_DECISION_MODE", "final")
    MULTIMODAL_CLIP_SECONDS = float(os.environ.get("MULTIMODAL_CLIP_SECONDS", "4.0"))
    MULTIMODAL_CLIP_STRIDE_SECONDS = float(os.environ.get("MULTIMODAL_CLIP_STRIDE_SECONDS", "2.0"))
    MULTIMODAL_FRAME_COUNT = int(os.environ.get("MULTIMODAL_FRAME_COUNT", "16"))
    MULTIMODAL_LABELS = os.environ.get(
        "MULTIMODAL_LABELS",
        "normal_driving,phone_interaction,phone_calling,in_vehicle_display_interaction,eating_distraction,drinking_distraction,visual_posture_distraction",
    )
    PHONE_DETECTOR_BACKEND = os.environ.get("PHONE_DETECTOR_BACKEND", "rtdetr_finetuned")
    YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", str(BASE_DIR / "models" / "yolo11n.pt"))
    YOLO_PRETRAINED_MODEL_PATH = os.environ.get("YOLO_PRETRAINED_MODEL_PATH", YOLO_MODEL_PATH)
    YOLO_FINETUNED_MODEL_PATH = os.environ.get(
        "YOLO_FINETUNED_MODEL_PATH",
        str(BASE_DIR / "models" / "phone_yolo26n.pt"),
    )
    RTDETR_MODEL_PATH = os.environ.get(
        "RTDETR_MODEL_PATH",
        str(BASE_DIR / "models" / "phone_rtdetr_l.pt"),
    )
    RTDETR_LIGHT_COMPARE_MODEL_PATH = os.environ.get(
        "RTDETR_LIGHT_COMPARE_MODEL_PATH",
        str(BASE_DIR / "models" / "phone_rtdetr_l_light_compare.pt"),
    )
    YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.35"))
    YOLO_SAMPLE_FPS = float(os.environ.get("YOLO_SAMPLE_FPS", "5.0"))
    REALTIME_CAMERA_INDEX = int(os.environ.get("REALTIME_CAMERA_INDEX", "0"))
    REALTIME_TARGET_FPS = float(os.environ.get("REALTIME_TARGET_FPS", "12.0"))
    REALTIME_BEHAVIOR_MODEL_BACKEND = os.environ.get("REALTIME_BEHAVIOR_MODEL_BACKEND", "mmrt_multimodal_full_v2_t4x2")
    REALTIME_ENABLE_YOLO = os.environ.get("REALTIME_ENABLE_YOLO", "0") == "1"
    REALTIME_YOLO_SAMPLE_FPS = float(os.environ.get("REALTIME_YOLO_SAMPLE_FPS", "1.0"))
    REALTIME_YOLO_CONFIDENCE_THRESHOLD = float(
        os.environ.get("REALTIME_YOLO_CONFIDENCE_THRESHOLD", os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.35"))
    )
    REALTIME_YOLO_MODEL_PATH = os.environ.get("REALTIME_YOLO_MODEL_PATH", str(BASE_DIR / "models" / "yolo11n.pt"))
    REALTIME_PHONE_DETECTOR_BACKEND = os.environ.get("REALTIME_PHONE_DETECTOR_BACKEND", "yolo_finetuned")
    REALTIME_ENABLE_SNAPSHOT_PREVIEW = os.environ.get("REALTIME_ENABLE_SNAPSHOT_PREVIEW", "1") == "1"
    REALTIME_SNAPSHOT_FPS = float(os.environ.get("REALTIME_SNAPSHOT_FPS", "3.0"))
    REALTIME_SNAPSHOT_WIDTH = int(os.environ.get("REALTIME_SNAPSHOT_WIDTH", "320"))
    REALTIME_ENABLE_EVENT_EVIDENCE = os.environ.get("REALTIME_ENABLE_EVENT_EVIDENCE", "1") == "1"
    EAR_THRESHOLD = float(os.environ.get("EAR_THRESHOLD", "0.23"))
    MAR_THRESHOLD = float(os.environ.get("MAR_THRESHOLD", "0.60"))
    EYE_CLOSURE_MIN_DURATION = float(os.environ.get("EYE_CLOSURE_MIN_DURATION", "1.0"))
    YAWN_MIN_DURATION = float(os.environ.get("YAWN_MIN_DURATION", "1.2"))
    LOOKING_AWAY_MIN_DURATION = float(os.environ.get("LOOKING_AWAY_MIN_DURATION", "1.5"))
    HEAD_DOWN_MIN_DURATION = float(os.environ.get("HEAD_DOWN_MIN_DURATION", "1.2"))
    FACE_MISSING_MIN_DURATION = float(os.environ.get("FACE_MISSING_MIN_DURATION", "2.0"))
    UPLOADED_SAMPLE_FPS = float(os.environ.get("UPLOADED_SAMPLE_FPS", "5.0"))
    PHONE_DRIVER_ROI = os.environ.get("PHONE_DRIVER_ROI", "0,0,1,1")
    PHONE_TEMPORAL_MIN_DETECTIONS = int(os.environ.get("PHONE_TEMPORAL_MIN_DETECTIONS", "2"))
    PHONE_TEMPORAL_MERGE_GAP_SECONDS = float(os.environ.get("PHONE_TEMPORAL_MERGE_GAP_SECONDS", "2.0"))
    VITDD_MODEL_PATH = os.environ.get("VITDD_MODEL_PATH", str(BASE_DIR / "models" / "vitdd_behavior.pt"))
    VITDD_MODEL_NAME = os.environ.get("VITDD_MODEL_NAME", "vit_base_patch16_224")
    VITDD_IMAGE_SIZE = int(os.environ.get("VITDD_IMAGE_SIZE", "224"))
    VITDD_SAMPLE_FPS = float(os.environ.get("VITDD_SAMPLE_FPS", "3.0"))
    VITDD_LABELS = os.environ.get(
        "VITDD_LABELS",
        "safe driving,texting right,talking phone right,texting left,talking phone left,"
        "operating radio,drinking,reaching behind,hair and makeup,talking to passenger",
    )
    LOG_DIR = os.environ.get("LOG_DIR", str(BASE_DIR / "logs"))
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


class DevelopmentConfig(Config):
    ENV = "development"


class ProductionConfig(Config):
    ENV = "production"
    DEBUG = False


def get_config():
    config_name = os.environ.get("APP_CONFIG", "development").lower()
    if config_name == "production":
        return ProductionConfig
    return DevelopmentConfig
