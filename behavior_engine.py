import math
from collections import defaultdict, deque
import config

class BehaviorEngine:
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=config.TEMPORAL_WINDOW_FRAMES * 2))

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def process_frame_data(self, persons, phones, poses):
        frame_analysis = []

        for person in persons:
            track_id = person["track_id"]
            if track_id is None:
                continue

            p_box = person["box"]
            p_center = person["center"]

            has_phone = False
            for phone in phones:
                dist = self._euclidean_distance(p_center, phone["center"])
                if dist < config.PHONE_ASSOCIATION_THRESHOLD:
                    has_phone = True
                    break

            head_dir = "CENTER"
            for pose in poses:
                if pose["box"] is not None:
                    p_x1, p_y1, p_x2, p_y2 = p_box
                    b_x1, b_y1, b_x2, b_y2 = pose["box"]
                    if abs(p_x1 - b_x1) < 50 and abs(p_y1 - b_y1) < 50:
                        head_dir = pose["head_direction"]
                        break

            self.history[track_id].append({
                "has_phone": has_phone,
                "head_dir": head_dir
            })

            recent_history = list(self.history[track_id])[-config.TEMPORAL_WINDOW_FRAMES:]
            phone_count = sum(1 for f in recent_history if f["has_phone"])
            looking_aside_count = sum(1 for f in recent_history if f["head_dir"] in ["LEFT", "RIGHT"])

            risk_score = 0
            detected_rules = []

            if has_phone:
                risk_score += config.RISK_WEIGHTS["phone_detected"]
                detected_rules.append("Phát hiện điện thoại")

            if phone_count >= config.TEMPORAL_WINDOW_FRAMES * 0.7:
                risk_score += config.RISK_WEIGHTS["phone_sustained"]
                detected_rules.append("Sử dụng điện thoại kéo dài")

            if head_dir in ["LEFT", "RIGHT"]:
                risk_score += config.RISK_WEIGHTS["looking_aside"]
                detected_rules.append(f"Quay đầu sang {head_dir}")

            if looking_aside_count >= config.TEMPORAL_WINDOW_FRAMES * 0.7:
                risk_score += config.RISK_WEIGHTS["looking_aside_sustained"]
                detected_rules.append("Nhìn lén/Quay đầu kéo dài (Mức 2)")

            risk_level = "LOW"
            if risk_score >= config.RISK_LEVEL_HIGH:
                risk_level = "HIGH"
            elif risk_score >= config.RISK_LEVEL_MEDIUM:
                risk_level = "MEDIUM"

            frame_analysis.append({
                "track_id": track_id,
                "box": p_box,
                "has_phone": has_phone,
                "head_dir": head_dir,
                "risk_score": min(risk_score, 100),
                "risk_level": risk_level,
                "rules": detected_rules
            })

        return frame_analysis
