import numpy as np
from ultralytics import YOLO
import config


class ObjectDetector:

    def __init__(self, model_path=config.YOLO_DETECTION_MODEL):
        self.model = YOLO(model_path)

    def detect_phones(self, frame):
        results = self.model(
             frame,
            verbose=False,
            imgsz=config.PHONE_IMGSZ,      
            conf=config.PHONE_CONF_THRESHOLD,
            iou=config.PHONE_IOU,          
            max_det=10,
            classes=[config.PHONE_CLASS_ID],
        )[0]

        phones = []
        if results.boxes is None or len(results.boxes) == 0:
            return phones

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            phones.append({
                "box": [x1, y1, x2, y2],
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "conf": float(conf),
            })

        return phones

