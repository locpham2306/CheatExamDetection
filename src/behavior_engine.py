from collections import defaultdict, deque
import config


class BehaviorEngine:
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=config.TEMPORAL_WINDOW_FRAMES * 2))
        self.last_seen = {}
        self.frame_index = 0

    @staticmethod
    def _normalized_center_distance(person, phone):
        px, py = person["center"]
        fx, fy = phone["center"]
        x1, y1, x2, y2 = person["box"]
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        return ((px - fx) / width) ** 2 + ((py - fy) / height) ** 2

    @staticmethod
    def _phone_is_near_person(person, phone):
        x1, y1, x2, y2 = person["box"]
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        margin_x = width * config.PHONE_ASSOCIATION_MARGIN_RATIO
        margin_y = height * config.PHONE_ASSOCIATION_MARGIN_RATIO
        phone_x, phone_y = phone["center"]
        return (
            x1 - margin_x <= phone_x <= x2 + margin_x
            and y1 - margin_y <= phone_y <= y2 + margin_y
        )

    def _associate_phones(self, persons, phones):
        """Gán mỗi điện thoại cho tối đa một người gần nhất."""
        assigned_track_ids = set()
        trackable_persons = [p for p in persons if p["track_id"] is not None]

        for phone in phones:
            candidates = [
                person
                for person in trackable_persons
                if self._phone_is_near_person(person, phone)
            ]
            if candidates:
                closest = min(
                    candidates,
                    key=lambda person: self._normalized_center_distance(person, phone),
                )
                assigned_track_ids.add(closest["track_id"])

        return assigned_track_ids

    def _remove_stale_histories(self):
        stale_ids = [
            track_id
            for track_id, last_frame in self.last_seen.items()
            if self.frame_index - last_frame > config.TRACK_HISTORY_TTL_FRAMES
        ]
        for track_id in stale_ids:
            self.last_seen.pop(track_id, None)
            self.history.pop(track_id, None)

    def process_frame_data(self, persons, phones):
        self.frame_index += 1
        phone_track_ids = self._associate_phones(persons, phones)
        frame_analysis = []

        for person in persons:
            track_id = person["track_id"]
            if track_id is None:
                continue

            p_box = person["box"]
            head_dir = person["head_direction"]
            has_phone = track_id in phone_track_ids
            self.last_seen[track_id] = self.frame_index

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

        self._remove_stale_histories()
        return frame_analysis

