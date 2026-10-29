from ultralytics import YOLO
import config


class PoseAnalyzer:
    """Dùng yolov8n-pose: detect người + tracking + hướng đầu."""

    def __init__(self, model_path=config.YOLO_POSE_MODEL):
        self.model = YOLO(model_path)

    def _head_direction(self, kpts, kpt_confs=None):
        # COCO keypoints: 0=nose, 1=l_eye, 2=r_eye, 3=l_ear, 4=r_ear
        if len(kpts) < 5:
            return "CENTER"

        if kpt_confs is not None:
            required = (kpt_confs[0], kpt_confs[3], kpt_confs[4])
            if any(conf < config.KEYPOINT_CONF_THRESHOLD for conf in required):
                return "CENTER"

        nose, l_ear, r_ear = kpts[0], kpts[3], kpts[4]
        if nose[0] > 0 and l_ear[0] > 0 and r_ear[0] > 0:
            ear_center_x = (l_ear[0] + r_ear[0]) / 2
            ear_dist = abs(l_ear[0] - r_ear[0]) + 1e-6
            offset = (nose[0] - ear_center_x) / ear_dist

            if offset > config.HEAD_TURN_THRESHOLD:
                return "RIGHT"
            if offset < -config.HEAD_TURN_THRESHOLD:
                return "LEFT"
        return "CENTER"

    def analyze_pose(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
            conf=config.PERSON_CONF_THRESHOLD,
            classes=[config.POSE_PERSON_CLASS_ID],
        )[0]

        persons = []
        if results.boxes is None or len(results.boxes) == 0 or results.keypoints is None:
            return persons

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        track_ids = (
            results.boxes.id.cpu().numpy().astype(int)
            if results.boxes.id is not None
            else [None] * len(boxes)
        )
        keypoints = results.keypoints.xyn.cpu().numpy()
        keypoint_confs = (
            results.keypoints.conf.cpu().numpy()
            if results.keypoints.conf is not None
            else [None] * len(boxes)
        )

        for box, conf, track_id, kpts, kpt_confs in zip(
            boxes, confs, track_ids, keypoints, keypoint_confs
        ):
            x1, y1, x2, y2 = box
            persons.append({
                "box": [x1, y1, x2, y2],
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "conf": float(conf),
                "track_id": track_id,
                "keypoints": kpts,
                "head_direction": self._head_direction(kpts, kpt_confs),
            })

        return persons
