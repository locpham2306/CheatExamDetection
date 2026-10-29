import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

YOLO_DETECTION_MODEL = os.path.join(MODEL_DIR, "best.pt")
YOLO_POSE_MODEL = os.path.join(MODEL_DIR, "yolov8n-pose.pt")

PHONE_CLASS_ID = 0
POSE_PERSON_CLASS_ID = 0

PHONE_IMGSZ = 960
PHONE_IOU = 0.5

PHONE_CONF_THRESHOLD = 0.3
PERSON_CONF_THRESHOLD = 0.5

PHONE_ASSOCIATION_MARGIN_RATIO = 0.2
HEAD_TURN_THRESHOLD = 0.25
KEYPOINT_CONF_THRESHOLD = 0.3
TEMPORAL_WINDOW_FRAMES = 15
TRACK_HISTORY_TTL_FRAMES = 60

RISK_WEIGHTS = {
    "phone_detected": 40,
    "phone_sustained": 30,
    "looking_aside": 15,
    "looking_aside_sustained": 25,
}

RISK_LEVEL_MEDIUM = 40
RISK_LEVEL_HIGH = 70
