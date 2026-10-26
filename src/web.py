import subprocess
import threading
import time

import cv2
import pandas as pd
import streamlit as st

import config
from behavior_engine import BehaviorEngine
from detector import ObjectDetector
from pose_analyzer import PoseAnalyzer


st.set_page_config(page_title="Exam Cheat Detection", layout="wide")
st.title("Phát hiện gian lận bằng camera ")

RISK_COLORS = {"LOW": (0, 255, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}



@st.cache_resource
def load_models():
    # best.pt -> điện thoại | yolov8n-pose -> người + tracking + hướng đầu
    return ObjectDetector(), PoseAnalyzer()


def reset_tracker(pose_analyzer):

    pose_analyzer.model.predictor = None


def run_adb(args):
    try:
        r = subprocess.run(["adb"] + args, capture_output=True, text=True, timeout=10)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except FileNotFoundError:
        return False, "Không tìm thấy lệnh `adb`. Hãy cài Android platform-tools và thêm vào PATH."
    except subprocess.TimeoutExpired:
        return False, "Lệnh adb bị quá thời gian chờ."


class LatestFrameReader:
    """Đọc camera ở luồng riêng và luôn giữ frame mới nhất, tránh bị trễ tích lũy
    khi việc xử lý YOLO chậm hơn tốc độ stream."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.failed = False
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def is_opened(self):
        return self.cap.isOpened()

    def start(self):
        self.thread.start()
        return self

    def _loop(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                self.failed = True
                break
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            frame, self.frame = self.frame, None
        return frame

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


def draw_frame(frame, phones, results):
    for phone in phones:
        x1, y1, x2, y2 = map(int, phone["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"PHONE {phone['conf']:.2f}", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for r in results:
        x1, y1, x2, y2 = map(int, r["box"])
        color = RISK_COLORS[r["risk_level"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID #{r['track_id']} | Risk {r['risk_score']}% | {r['head_dir']}"
        cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if r["rules"]:
            cv2.putText(frame, " | ".join(r["rules"]), (x1, min(frame.shape[0] - 10, y2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def update_stats(stats, results):
    for r in results:
        s = stats.setdefault(int(r["track_id"]), {
            "Số frame": 0,
            "Frame có điện thoại": 0,
            "Frame quay đầu": 0,
            "Frame nguy cơ cao": 0,
            "Điểm rủi ro cao nhất": 0,
        })
        s["Số frame"] += 1
        s["Frame có điện thoại"] += int(r["has_phone"])
        s["Frame quay đầu"] += int(r["head_dir"] in ("LEFT", "RIGHT"))
        s["Frame nguy cơ cao"] += int(r["risk_level"] == "HIGH")
        s["Điểm rủi ro cao nhất"] = max(s["Điểm rủi ro cao nhất"], r["risk_score"])


def render_stats(placeholder, stats):
    if not stats:
        placeholder.info("Chưa có dữ liệu.")
        return
    df = pd.DataFrame.from_dict(stats, orient="index")
    df = df.sort_values("Điểm rủi ro cao nhất", ascending=False)

    df = pd.DataFrame({
        "Phát hiện điện thoại": raw["Frame có điện thoại"].gt(0).map({True: "Có", False: "Không"}),
        "Phát hiện quay đầu": raw["Frame quay đầu"].gt(0).map({True: "Có", False: "Không"}),
        "Điểm nguy cơ": raw["Điểm rủi ro cao nhất"],
    })
    df.index.name = "ID"

    def highlight_high_risk(row):
        if row["Điểm rủi ro cao nhất"] >= config.RISK_LEVEL_HIGH:
            return ["background-color: #ff4b4b; color: white; font-weight: bold"] * len(row)
        return [""] * len(row)

    placeholder.dataframe(df.style.apply(highlight_high_risk, axis=1),
                          use_container_width=True)


if "running" not in st.session_state:
    st.session_state.running = False
if "stats" not in st.session_state:
    st.session_state.stats = {}

detector, pose_analyzer = load_models()

with st.sidebar:
    st.header("Nguồn camera")
    mode = st.radio("Chọn nguồn", ["Điện thoại qua ADB", "Nguồn khác (URL / số camera)"])

    if mode == "Điện thoại qua ADB":
        port = st.number_input("Cổng (port)", 1024, 65535, 8080,
                               help="IP Webcam dùng 8080, DroidCam dùng 4747.")
        path = st.text_input("Đường dẫn stream", "/video")
        source = f"http://127.0.0.1:{int(port)}{path}"
        st.code(source)

        if st.button("Kiểm tra thiết bị ADB"):
            ok, msg = run_adb(["devices"])
            (st.success if ok else st.error)(msg)
    else:
        raw = st.text_input("URL stream hoặc số camera", "0")
        source = int(raw) if raw.strip().isdigit() else raw.strip()

    st.header("Hiệu năng")
    process_every_n = st.slider("Xử lý mỗi N frame", 1, 5, 2,
                                help="Số lớn hơn = chạy nhanh hơn nhưng bỏ qua một số frame.")
    max_width = st.select_slider("Độ rộng tối đa khi xử lý (px)", options=[480, 640, 960, 1280], value=960,
                                 help="Hạ xuống để chạy nhanh hơn trên máy không có GPU.")

col_a, col_b = st.columns(2)
if col_a.button("▶️ Bắt đầu", type="primary", use_container_width=True):
    st.session_state.running = True
    st.session_state.stats = {}
if col_b.button("⏹️ Dừng", use_container_width=True):
    st.session_state.running = False


col_video, col_stats = st.columns([3, 2], gap="large")

with col_video:
    st.subheader("Video giám sát")
    status = st.empty()
    frame_placeholder = st.empty()

with col_stats:
    st.subheader("Tổng kết theo từng ID")
    stats_placeholder = st.empty()

if st.session_state.running:
    if mode == "Điện thoại qua ADB":
        ok, msg = run_adb(["forward", f"tcp:{int(port)}", f"tcp:{int(port)}"])
        if not ok:
            st.session_state.running = False
            status.error(f"adb forward thất bại: {msg}")
            st.stop()

    reader = LatestFrameReader(source)
    if not reader.is_opened():
        reader.release()
        st.session_state.running = False
        status.error(
            "Không mở được luồng camera. Hãy kiểm tra: app trên điện thoại đã bấm "
            "'Start server' chưa, cáp USB, USB debugging, và cổng/đường dẫn stream."
        )
        st.stop()
    reader.start()

    behavior_engine = BehaviorEngine()
    reset_tracker(pose_analyzer)

    stats = st.session_state.stats
    last_phones, last_results = [], []
    frame_idx = 0
    fps_count, fps_time, fps = 0, time.time(), 0.0
    last_frame_time = time.time()

    try:
        while True:
            frame = reader.read()
            if frame is None:
                if reader.failed or time.time() - last_frame_time > 10:
                    status.error("Mất kết nối với camera điện thoại.")
                    break
                time.sleep(0.005)
                continue
            last_frame_time = time.time()

            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(h * max_width / w)))

            frame_idx += 1
            if frame_idx % process_every_n == 0:
                last_phones = detector.detect_phones(frame)
                persons = pose_analyzer.analyze_pose(frame)
                last_results = behavior_engine.process_frame_data(persons, last_phones)
                update_stats(stats, last_results)

            frame = draw_frame(frame, last_phones, last_results)
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_count / (time.time() - fps_time)
                fps_count, fps_time = 0, time.time()
                render_stats(stats_placeholder, stats)
                status.success("Đang chạy trực tiếp 🔴")
    finally:
        reader.release()
        st.session_state.running = False

render_stats(stats_placeholder, st.session_state.stats)