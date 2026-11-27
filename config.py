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
    # Research/demo routes are retained for reproducibility, but they are not
    # part of the product runtime unless explicitly enabled.
    ENABLE_LEGACY_RESEARCH_ROUTES = os.environ.get(
        "ENABLE_LEGACY_RESEARCH_ROUTES", "0"
    ) == "1"
    # Driver Safety v1 is the active product pipeline. Legacy multimodal
    # artifacts remain on disk only for historical sessions and reports.
    DRIVER_SAFETY_MODEL_DIR = os.environ.get(
        "DRIVER_SAFETY_MODEL_DIR", str(BASE_DIR / "models" / "driver_safety_v1")
    )
    DRIVER_SAFETY_FACE_DETECTOR_PATH = os.environ.get(
        "DRIVER_SAFETY_FACE_DETECTOR_PATH", str(Path(DRIVER_SAFETY_MODEL_DIR) / "face_detection.pt")
    )
    DRIVER_SAFETY_FACE_VERIFIER_PATH = os.environ.get(
        "DRIVER_SAFETY_FACE_VERIFIER_PATH", str(Path(DRIVER_SAFETY_MODEL_DIR) / "face_verification.h5")
    )
    DRIVER_SAFETY_FACE_VERIFY_THRESHOLD = float(os.environ.get("DRIVER_SAFETY_FACE_VERIFY_THRESHOLD", "0.94"))
    DRIVER_SAFETY_FACE_ENROLLMENT_DIR = os.environ.get(
        "DRIVER_SAFETY_FACE_ENROLLMENT_DIR", str(BASE_DIR / "uploads" / "face_enrollment")
    )
    DRIVER_SAFETY_DROWSINESS_MODEL_PATH = os.environ.get(
        "DRIVER_SAFETY_DROWSINESS_MODEL_PATH", str(Path(DRIVER_SAFETY_MODEL_DIR) / "best_model.h5")
    )
    DRIVER_SAFETY_DROWSINESS_OPENVINO_PATH = os.environ.get(
        "DRIVER_SAFETY_DROWSINESS_OPENVINO_PATH",
        str(Path(DRIVER_SAFETY_MODEL_DIR) / "openvino" / "drowsiness.xml"),
    )
    DRIVER_SAFETY_FACE_VERIFIER_OPENVINO_PATH = os.environ.get(
        "DRIVER_SAFETY_FACE_VERIFIER_OPENVINO_PATH",
        str(Path(DRIVER_SAFETY_MODEL_DIR) / "openvino" / "face_verification.xml"),
    )
    DRIVER_SAFETY_SCALER_PATH = os.environ.get(
        "DRIVER_SAFETY_SCALER_PATH", str(Path(DRIVER_SAFETY_MODEL_DIR) / "scaler.joblib")
    )
    # Values below come from the report and its original webcam inference code.
    DRIVER_SAFETY_FACE_CONFIDENCE = float(os.environ.get("DRIVER_SAFETY_FACE_CONFIDENCE", "0.251"))
    DRIVER_SAFETY_FACE_IMGSZ = int(os.environ.get("DRIVER_SAFETY_FACE_IMGSZ", "640"))
    DRIVER_SAFETY_FACE_DETECTOR_FPS = float(os.environ.get("DRIVER_SAFETY_FACE_DETECTOR_FPS", "2.0"))
    DRIVER_SAFETY_FACE_DEVICE = os.environ.get("DRIVER_SAFETY_FACE_DEVICE", "auto")
    DRIVER_SAFETY_SEQUENCE_LENGTH = int(os.environ.get("DRIVER_SAFETY_SEQUENCE_LENGTH", "15"))
    DRIVER_SAFETY_INFERENCE_STRIDE = int(os.environ.get("DRIVER_SAFETY_INFERENCE_STRIDE", "2"))
    DRIVER_SAFETY_FEATURE_GAP_RESET_SECONDS = float(
        # Inference latency must not erase a valid temporal clip.
        os.environ.get("DRIVER_SAFETY_FEATURE_GAP_RESET_SECONDS", "2.0")
    )
    DRIVER_SAFETY_DECISION_THRESHOLD = float(os.environ.get("DRIVER_SAFETY_DECISION_THRESHOLD", "0.5"))
    DRIVER_SAFETY_ALERT_THRESHOLD = float(os.environ.get("DRIVER_SAFETY_ALERT_THRESHOLD", "0.7"))
    DRIVER_SAFETY_EVENT_CONFIRM_SECONDS = float(os.environ.get("DRIVER_SAFETY_EVENT_CONFIRM_SECONDS", "3.0"))
    DRIVER_ACTION_MODEL_DIR = os.environ.get(
        "DRIVER_ACTION_MODEL_DIR",
        str(BASE_DIR / "models" / "openvino_driver_action_adas_0002" / "FP16-INT8"),
    )
    DRIVER_ACTION_ENCODER_PATH = os.environ.get(
        "DRIVER_ACTION_ENCODER_PATH",
        str(Path(DRIVER_ACTION_MODEL_DIR) / "driver-action-recognition-adas-0002-encoder.xml"),
    )
    DRIVER_ACTION_DECODER_PATH = os.environ.get(
        "DRIVER_ACTION_DECODER_PATH",
        str(Path(DRIVER_ACTION_MODEL_DIR) / "driver-action-recognition-adas-0002-decoder.xml"),
    )
    DRIVER_ACTION_DEVICE = os.environ.get("DRIVER_ACTION_DEVICE", "auto")
    DRIVER_ACTION_WINDOW_SIZE = int(os.environ.get("DRIVER_ACTION_WINDOW_SIZE", "16"))
    DRIVER_ACTION_SAMPLE_STRIDE_FRAMES = int(os.environ.get("DRIVER_ACTION_SAMPLE_STRIDE_FRAMES", "2"))
    DRIVER_ACTION_INFERENCE_STRIDE = int(os.environ.get("DRIVER_ACTION_INFERENCE_STRIDE", "1"))
    DRIVER_ACTION_LABEL_SMOOTHING_FRAMES = int(
        os.environ.get("DRIVER_ACTION_LABEL_SMOOTHING_FRAMES", "15")
    )
    DRIVER_ACTION_CONFIDENCE_THRESHOLD = float(os.environ.get("DRIVER_ACTION_CONFIDENCE_THRESHOLD", "0.45"))
    # The action model is prone to confusing neutral cabin posture with its
    # eating/drinking class. Require stronger evidence for this class only.
    DRIVER_ACTION_DRINKING_EATING_THRESHOLD = float(
        os.environ.get("DRIVER_ACTION_DRINKING_EATING_THRESHOLD", "0.70")
    )
    # Keep the model probability intact, but give this easily-confused class
    # a lower contribution to the business risk recommendation.
    DRIVER_ACTION_DRINKING_EATING_RISK_MULTIPLIER = float(
        os.environ.get("DRIVER_ACTION_DRINKING_EATING_RISK_MULTIPLIER", "0.5")
    )
    DRIVER_ACTION_EVENT_CONFIRM_SECONDS = float(os.environ.get("DRIVER_ACTION_EVENT_CONFIRM_SECONDS", "2.0"))
    MULTIMODAL_MODEL_BACKEND = os.environ.get("MULTIMODAL_MODEL_BACKEND", "driver_safety_v1")
    MULTIMODAL_MODEL_PATH = os.environ.get(
        "MULTIMODAL_MODEL_PATH",
        str(BASE_DIR / "models" / "mmrt_artifacts_bundle"),
    )
    MMRT_V2_MODEL_PATH = os.environ.get(
        "MMRT_V2_MODEL_PATH",
        str(BASE_DIR / "models" / "mmrt_artifacts_bundle"),
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
    PHONE_DETECTOR_BACKEND = os.environ.get("PHONE_DETECTOR_BACKEND", "disabled")
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
    YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.35"))
    YOLO_SAMPLE_FPS = float(os.environ.get("YOLO_SAMPLE_FPS", "5.0"))
    REALTIME_CAMERA_INDEX = int(os.environ.get("REALTIME_CAMERA_INDEX", "0"))
    REALTIME_CAMERA_WIDTH = int(os.environ.get("REALTIME_CAMERA_WIDTH", "640"))
    REALTIME_CAMERA_HEIGHT = int(os.environ.get("REALTIME_CAMERA_HEIGHT", "480"))
    REALTIME_TARGET_FPS = float(os.environ.get("REALTIME_TARGET_FPS", "30.0"))
    REALTIME_IDLE_FEED_FPS = float(os.environ.get("REALTIME_IDLE_FEED_FPS", "12.0"))
    REALTIME_PREVIEW_MAX_FPS = float(os.environ.get("REALTIME_PREVIEW_MAX_FPS", "12.0"))
    REALTIME_PREVIEW_WIDTH = int(os.environ.get("REALTIME_PREVIEW_WIDTH", "640"))
    REALTIME_PREVIEW_JPEG_QUALITY = int(os.environ.get("REALTIME_PREVIEW_JPEG_QUALITY", "72"))
    REALTIME_BEHAVIOR_MODEL_BACKEND = os.environ.get("REALTIME_BEHAVIOR_MODEL_BACKEND", "driver_safety_v1")
    REALTIME_ENABLE_YOLO = os.environ.get("REALTIME_ENABLE_YOLO", "0") == "1"
    REALTIME_YOLO_SAMPLE_FPS = float(os.environ.get("REALTIME_YOLO_SAMPLE_FPS", "1.0"))
    REALTIME_YOLO_CONFIDENCE_THRESHOLD = float(
        os.environ.get("REALTIME_YOLO_CONFIDENCE_THRESHOLD", os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.35"))
    )
    REALTIME_YOLO_MODEL_PATH = os.environ.get("REALTIME_YOLO_MODEL_PATH", str(BASE_DIR / "models" / "yolo11n.pt"))
    # Realtime phone decisions currently come from the action-recognition
    # model. Keep the standalone detector available for separate testing, but
    # do not mix its signal into the main behavior demo.
    REALTIME_PHONE_DETECTOR_BACKEND = os.environ.get("REALTIME_PHONE_DETECTOR_BACKEND", "disabled")
    REALTIME_PHONE_CONTEXT_FILTER = os.environ.get("REALTIME_PHONE_CONTEXT_FILTER", "1") == "1"
    REALTIME_ENABLE_SNAPSHOT_PREVIEW = os.environ.get("REALTIME_ENABLE_SNAPSHOT_PREVIEW", "1") == "1"
    REALTIME_SNAPSHOT_FPS = float(os.environ.get("REALTIME_SNAPSHOT_FPS", "3.0"))
    REALTIME_SNAPSHOT_WIDTH = int(os.environ.get("REALTIME_SNAPSHOT_WIDTH", "320"))
    REALTIME_ENABLE_EVENT_EVIDENCE = os.environ.get("REALTIME_ENABLE_EVENT_EVIDENCE", "1") == "1"
    EAR_THRESHOLD = float(os.environ.get("EAR_THRESHOLD", "0.23"))
    MAR_THRESHOLD = float(os.environ.get("MAR_THRESHOLD", "0.60"))
    EYE_CLOSURE_MIN_DURATION = float(os.environ.get("EYE_CLOSURE_MIN_DURATION", "1.0"))
    YAWN_MIN_DURATION = float(os.environ.get("YAWN_MIN_DURATION", "2.0"))
    LOOKING_AWAY_MIN_DURATION = float(os.environ.get("LOOKING_AWAY_MIN_DURATION", "2.0"))
    HEAD_DOWN_MIN_DURATION = float(os.environ.get("HEAD_DOWN_MIN_DURATION", "2.0"))
    FACE_LOOKING_AWAY_OFFSET_THRESHOLD = float(
        os.environ.get("FACE_LOOKING_AWAY_OFFSET_THRESHOLD", "0.07")
    )
    # Looking-away is decided from calibrated head yaw, not instantaneous
    # motion. The offset threshold above remains as a compatibility fallback.
    FACE_LOOKING_AWAY_YAW_THRESHOLD_DEGREES = float(
        os.environ.get("FACE_LOOKING_AWAY_YAW_THRESHOLD_DEGREES", "20.0")
    )
    FACE_LOOKING_AWAY_YAW_RELEASE_DEGREES = float(
        os.environ.get("FACE_LOOKING_AWAY_YAW_RELEASE_DEGREES", "15.0")
    )
    FACE_HEAD_DOWN_OFFSET_THRESHOLD = float(
        os.environ.get("FACE_HEAD_DOWN_OFFSET_THRESHOLD", "0.08")
    )
    FACE_BEHAVIOR_CALIBRATION_SAMPLES = int(
        os.environ.get("FACE_BEHAVIOR_CALIBRATION_SAMPLES", "20")
    )
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
