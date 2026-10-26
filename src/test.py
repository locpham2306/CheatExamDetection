import os
from ultralytics import YOLO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) 
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
YOLO_DETECTION_MODEL = os.path.join(MODEL_DIR, "best.pt")
model = YOLO(YOLO_DETECTION_MODEL)
print(model.names)