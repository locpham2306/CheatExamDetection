import os

# Model Paths (Mặc định dùng weights của Ultralytics, có thể thay bằng 'best.pt' sau khi fine-tune)
YOLO_DETECTION_MODEL = "yolov8n.pt"
YOLO_POSE_MODEL = "yolov8n-pose.pt"

# Object Class IDs (COCO Dataset)
PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67

# Spatial & Temporal Thresholds
PHONE_ASSOCIATION_THRESHOLD = 150  # Khoảng cách tối đa (pixels) giữa tay/thân người và điện thoại
HEAD_TURN_THRESHOLD = 0.25         # Ngưỡng lệch mũi để xác định quay đầu trái/phải
TEMPORAL_WINDOW_FRAMES = 15        # Số frame tối thiểu duy trì hành vi để kích hoạt cảnh báo

# Risk Scoring Engine Rules
RISK_WEIGHTS = {
    "phone_detected": 40,
    "phone_sustained": 30,
    "looking_aside": 15,
    "looking_aside_sustained": 25,
}

RISK_LEVEL_MEDIUM = 40
RISK_LEVEL_HIGH = 70
