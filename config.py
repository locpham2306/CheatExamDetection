import os

YOLO_DETECTION_MODEL = "yolov8n.pt"
YOLO_POSE_MODEL = "yolov8n-pose.pt"

PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67


PHONE_ASSOCIATION_THRESHOLD = 150  
HEAD_TURN_THRESHOLD = 0.25       
TEMPORAL_WINDOW_FRAMES = 15        

RISK_WEIGHTS = {
    "phone_detected": 40,
    "phone_sustained": 30,
    "looking_aside": 15,
    "looking_aside_sustained": 25,
}

RISK_LEVEL_MEDIUM = 40
RISK_LEVEL_HIGH = 70
