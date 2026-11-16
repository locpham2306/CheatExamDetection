import numpy as np
from ultralytics import YOLO
import config

POSE_IMGSZ = getattr(config, "POSE_IMGSZ", 1280)
MAX_PERSONS = getattr(config, "MAX_PERSONS", 60)
USE_TRACKER = getattr(config, "USE_TRACKER", False)      # False: dùng SeatIDAssigner để gán ID
TRACKER_CFG = getattr(config, "TRACKER_CFG", "bytetrack.yaml")
# Ngưỡng tin cậy tối thiểu của keypoint đầu/vai. Đặt thấp (0.3) để không bỏ sót người
# bị nghiêng/che; trước đây 0.5 làm nhiều người bị trả về CENTER.
KPT_CONF_THRESHOLD = getattr(config, "KPT_CONF_THRESHOLD", 0.3)

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 1, 2, 3, 4, 5, 6


class PoseAnalyzer:
    """Dùng yolov8m-pose: detect người + độ lệch hướng đầu (head_offset).
    Mặc định KHÔNG tracking (USE_TRACKER=False) vì ID được gán theo chỗ ngồi."""

    def __init__(self, model_path=config.YOLO_POSE_MODEL):
        self.model = YOLO(model_path)

    # ---------- hướng đầu ----------
    @staticmethod
    def _visible(kpts, kconf, idx):
        if kconf is not None:
            return kconf[idx] >= KPT_CONF_THRESHOLD
        return kpts[idx][0] > 0

    def _head_offset(self, kpts, kconf=None):
        """Độ lệch mũi so với tâm đầu, chuẩn hoá về cùng thang với 'mũi so với 2 tai'.
        Dùng lần lượt: 2 tai -> 2 mắt -> 2 vai, tuỳ keypoint nào nhìn thấy.
        Trả về None nếu không đủ dữ liệu."""
        if len(kpts) < 7 or not self._visible(kpts, kconf, NOSE):
            return None
        nose_x = kpts[NOSE][0]

        def ratio(a, b, scale):
            dist = abs(kpts[a][0] - kpts[b][0])
            if dist < 1e-4:
                return None
            center = (kpts[a][0] + kpts[b][0]) / 2
            return float(np.clip(scale * (nose_x - center) / dist, -3, 3))

        if self._visible(kpts, kconf, L_EAR) and self._visible(kpts, kconf, R_EAR):
            return ratio(L_EAR, R_EAR, 1.0)
        if self._visible(kpts, kconf, L_EYE) and self._visible(kpts, kconf, R_EYE):
            return ratio(L_EYE, R_EYE, 0.5)
        if self._visible(kpts, kconf, L_SHOULDER) and self._visible(kpts, kconf, R_SHOULDER):
            return ratio(L_SHOULDER, R_SHOULDER, 2.5)
        return None

    @staticmethod
    def direction_from_offset(offset, baseline=0.0):
        """So độ lệch với mức 'bình thường' của từng chỗ ngồi (baseline) rồi mới kết luận."""
        if offset is None:
            return "CENTER"
        d = offset - baseline
        if d > config.HEAD_TURN_THRESHOLD:
            return "RIGHT"
        if d < -config.HEAD_TURN_THRESHOLD:
            return "LEFT"
        return "CENTER"

    # ---------- chạy model ----------
    def _run_model(self, frame):
        common = dict(
            verbose=False,
            imgsz=POSE_IMGSZ,
            conf=config.PERSON_CONF_THRESHOLD,
            classes=[config.POSE_PERSON_CLASS_ID],
            max_det=MAX_PERSONS,
        )
        if USE_TRACKER:
            return self.model.track(frame, persist=True, tracker=TRACKER_CFG, **common)[0]
        return self.model(frame, **common)[0]

    def analyze_pose(self, frame):
        results = self._run_model(frame)

        persons = []
        if results.boxes is None or len(results.boxes) == 0 or results.keypoints is None:
            return persons

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        if results.boxes.id is not None:
            track_ids = results.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [None] * len(boxes)

        keypoints = results.keypoints.xyn.cpu().numpy()
        kp_confs = (
            results.keypoints.conf.cpu().numpy()
            if results.keypoints.conf is not None
            else [None] * len(boxes)
        )

        for box, conf, track_id, kpts, kconf in zip(boxes, confs, track_ids, keypoints, kp_confs):
            x1, y1, x2, y2 = box
            offset = self._head_offset(kpts, kconf)
            persons.append({
                "box": [x1, y1, x2, y2],
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "conf": float(conf),
                "track_id": track_id,          # sẽ bị SeatIDAssigner ghi đè
                "keypoints": kpts,
                "head_offset": offset,
                "head_direction": self.direction_from_offset(offset),
            })

        return persons