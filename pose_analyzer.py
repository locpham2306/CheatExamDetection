import numpy as np
from ultralytics import YOLO
import config

class PoseAnalyzer:
    def __init__(self, model_path=config.YOLO_POSE_MODEL):
        self.model = YOLO(model_path)

    def analyze_pose(self, frame):
        """
        Ước lượng tư thế và hướng nhìn (Trái/Phải/Thẳng) dựa trên Keypoints COCO.
        Keypoints Index: 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
        """
        results = self.model(frame, verbose=False)[0]
        pose_results = []

        if results.keypoints is not None:
            keypoints_data = results.keypoints.xyn.cpu().numpy() # Normalized keypoints
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []

            for idx, kpts in enumerate(keypoints_data):
                head_direction = "CENTER"
                if len(kpts) >= 5:
                    nose = kpts[0]
                    l_ear = kpts[3]
                    r_ear = kpts[4]

                    # Nếu phát hiện đủ Mũi và 2 Tai
                    if nose[0] > 0 and l_ear[0] > 0 and r_ear[0] > 0:
                        ear_center_x = (l_ear[0] + r_ear[0]) / 2
                        ear_dist = abs(l_ear[0] - r_ear[0]) + 1e-6
                        
                        # Độ lệch của mũi so với trung điểm 2 tai
                        offset = (nose[0] - ear_center_x) / ear_dist

                        if offset > config.HEAD_TURN_THRESHOLD:
                            head_direction = "RIGHT"
                        elif offset < -config.HEAD_TURN_THRESHOLD:
                            head_direction = "LEFT"

                box = boxes[idx] if idx < len(boxes) else None
                pose_results.append({
                    "box": box,
                    "keypoints": kpts,
                    "head_direction": head_direction
                })

        return pose_results
