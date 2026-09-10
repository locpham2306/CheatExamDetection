import cv2
import numpy as np
from ultralytics import YOLO
import config

class ObjectDetector:
    def __init__(self, model_path=config.YOLO_DETECTION_MODEL):
        self.model = YOLO(model_path)

    def detect_and_track(self, frame):
        """
        Thực hiện Object Detection và ByteTrack Tracking trên frame.
        """
        results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
        
        persons = []
        phones = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            
            # Kiểm tra xem tracking IDs có khả dụng không
            track_ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else [None] * len(boxes)

            for box, cls, conf, track_id in zip(boxes, clss, confs, track_ids):
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                item = {
                    "box": [x1, y1, x2, y2],
                    "center": center,
                    "conf": conf,
                    "track_id": track_id
                }

                if cls == config.PERSON_CLASS_ID:
                    persons.append(item)
                elif cls == config.PHONE_CLASS_ID:
                    phones.append(item)

        return persons, phones
