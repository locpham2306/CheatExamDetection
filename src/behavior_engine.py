import math
from collections import defaultdict, deque
import config


class BehaviorEngine:
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=config.TEMPORAL_WINDOW_FRAMES * 2))

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def process_frame_data(self, persons, phones):
        frame_analysis = []

        for person in persons:
            track_id = person["track_id"]
            if track_id is None:
                continue

            p_box = person["box"]
            p_center = person["center"]
            head_dir = person["head_direction"]

            has_phone = any(
                self._euclidean_distance(p_center, phone["center"]) < config.PHONE_ASSOCIATION_THRESHOLD
                for phone in phones
            )

            self.history[track_id].append({
                "has_phone": has_phone,
                "head_dir": head_dir,
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
                "conf": person["conf"],
                "has_phone": has_phone,
                "head_dir": head_dir,
                "risk_score": min(risk_score, 100),
                "risk_level": risk_level,
                "rules": detected_rules,
            })

        return frame_analysis

