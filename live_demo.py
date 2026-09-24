import cv2

from behavior_engine import BehaviorEngine
from detector import ObjectDetector
from pose_analyzer import PoseAnalyzer


WINDOW_TITLE = "Exam Cheat Detection - LIVE (Q/ESC to quit)"
PROCESS_EVERY_N_FRAMES = 2


def draw_detections(frame, persons, phones, results):
    risk_by_track_id = {result["track_id"]: result for result in results}

    for person in persons:
        x1, y1, x2, y2 = map(int, person["box"])
        result = risk_by_track_id.get(person["track_id"])
        level = result["risk_level"] if result else "LOW"
        color = {"LOW": (0, 255, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}[level]
        label = f"PERSON {person['conf']:.2f}"
        if result:
            label += f" | RISK {result['risk_score']}% | {result['head_dir']}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for phone in phones:
        x1, y1, x2, y2 = map(int, phone["box"])
        color = (0, 0, 255)
        label = f"PHONE {phone['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    for result in results:
        if result["has_phone"]:
            x1, _, _, y2 = map(int, result["box"])
            cv2.putText(frame, "PHONE DETECTED", (x1, min(frame.shape[0] - 10, y2 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


def main():
    print("Loading YOLO models...")
    detector = ObjectDetector()
    pose_analyzer = PoseAnalyzer()
    behavior_engine = BehaviorEngine()

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Khong mo duoc camera. Hay kiem tra quyen Camera cua Windows.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    frame_number = 0
    latest_persons = []
    latest_phones = []
    latest_results = []
    print("Camera is running. Press Q or Esc to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Mat ket noi voi camera.")
            frame_number += 1
            if frame_number % PROCESS_EVERY_N_FRAMES == 0:
                latest_persons, latest_phones = detector.detect_and_track(frame)
                poses = pose_analyzer.analyze_pose(frame)
                latest_results = behavior_engine.process_frame_data(latest_persons, latest_phones, poses)

            draw_detections(frame, latest_persons, latest_phones, latest_results)
            cv2.putText(frame, "LIVE - Q/ESC TO EXIT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
