import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

import config
from behavior_engine import BehaviorEngine
from detector import ObjectDetector
from pose_analyzer import PoseAnalyzer


st.set_page_config(page_title="Exam Cheat Detection", layout="wide")
st.title("Phát hiện gian lận qua camera hoặc video")
st.caption(
    "Dùng camera điện thoại qua Wi-Fi, webcam máy tính hoặc tải video có sẵn lên. "
    "Không cần kết nối điện thoại bằng cáp USB."
)

RISK_COLORS = {
    "LOW": (0, 255, 0),
    "MEDIUM": (0, 165, 255),
    "HIGH": (0, 0, 255),
}


def load_models():
    # best.pt -> điện thoại | yolov8n-pose -> người + tracking + hướng đầu
    return ObjectDetector(), PoseAnalyzer()


def reset_tracker(pose_analyzer):
    pose_analyzer.model.predictor = None


class LatestFrameReader:
    """Luôn giữ frame mới nhất để luồng trực tiếp không bị trễ tích lũy."""

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
                if not self.stopped:
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
        self.cap.release()
        if self.thread.is_alive():
            self.thread.join(timeout=1)


def resize_frame(frame, max_width):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    new_height = int(height * max_width / width)
    return cv2.resize(frame, (max_width, new_height))


def draw_frame(frame, phones, results):
    for phone in phones:
        x1, y1, x2, y2 = map(int, phone["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            frame,
            f"PHONE {phone['conf']:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    for result in results:
        x1, y1, x2, y2 = map(int, result["box"])
        color = RISK_COLORS[result["risk_level"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (
            f"ID #{result['track_id']} | Risk {result['risk_score']}% "
            f"| {result['head_dir']}"
        )
        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        if result["rules"]:
            cv2.putText(
                frame,
                " | ".join(result["rules"]),
                (x1, min(frame.shape[0] - 10, y2 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
    return frame


def update_stats(stats, results):
    for result in results:
        student_stats = stats.setdefault(
            int(result["track_id"]),
            {
                "Số frame": 0,
                "Frame có điện thoại": 0,
                "Frame quay đầu": 0,
                "Frame nguy cơ cao": 0,
                "Điểm rủi ro cao nhất": 0,
            },
        )
        student_stats["Số frame"] += 1
        student_stats["Frame có điện thoại"] += int(result["has_phone"])
        student_stats["Frame quay đầu"] += int(
            result["head_dir"] in ("LEFT", "RIGHT")
        )
        student_stats["Frame nguy cơ cao"] += int(result["risk_level"] == "HIGH")
        student_stats["Điểm rủi ro cao nhất"] = max(
            student_stats["Điểm rủi ro cao nhất"], result["risk_score"]
        )


def render_stats(placeholder, stats):
    if not stats:
        placeholder.info("Chưa có dữ liệu.")
        return

    raw = pd.DataFrame.from_dict(stats, orient="index")
    raw = raw.sort_values("Điểm rủi ro cao nhất", ascending=False)
    table = pd.DataFrame(
        {
            "Phát hiện điện thoại": raw["Frame có điện thoại"]
            .gt(0)
            .map({True: "Có", False: "Không"}),
            "Phát hiện quay đầu": raw["Frame quay đầu"]
            .gt(0)
            .map({True: "Có", False: "Không"}),
            "Điểm nguy cơ": raw["Điểm rủi ro cao nhất"],
        }
    )
    table.index.name = "ID"

    def highlight_high_risk(row):
        if row["Điểm nguy cơ"] >= config.RISK_LEVEL_HIGH:
            style = "background-color: #ff4b4b; color: white; font-weight: bold"
            return [style] * len(row)
        return [""] * len(row)

    placeholder.dataframe(
        table.style.apply(highlight_high_risk, axis=1),
        use_container_width=True,
    )


def analyze_frame(frame, detector, pose_analyzer, behavior_engine):
    phones = detector.detect_phones(frame)
    persons = pose_analyzer.analyze_pose(frame)
    results = behavior_engine.process_frame_data(persons, phones)
    return phones, results


def run_live_source(
    source,
    detector,
    pose_analyzer,
    process_every_n,
    max_width,
    frame_placeholder,
    stats_placeholder,
    status,
):
    reader = LatestFrameReader(source)
    if not reader.is_opened():
        reader.release()
        st.session_state.active_mode = None
        status.error(
            "Không mở được camera. Hãy kiểm tra URL, kết nối Wi-Fi và đảm bảo "
            "ứng dụng camera trên điện thoại đã bật máy chủ video."
        )
        return

    reader.start()
    behavior_engine = BehaviorEngine()
    reset_tracker(pose_analyzer)
    stats = st.session_state.stats
    last_phones, last_results = [], []
    frame_index = 0
    fps_count, fps_started_at, display_fps = 0, time.time(), 0.0
    last_frame_at = time.time()

    try:
        while st.session_state.active_mode == "live":
            frame = reader.read()
            if frame is None:
                if reader.failed or time.time() - last_frame_at > 10:
                    status.error("Mất kết nối với nguồn camera.")
                    break
                time.sleep(0.005)
                continue

            last_frame_at = time.time()
            frame = resize_frame(frame, max_width)
            frame_index += 1

            if (frame_index - 1) % process_every_n == 0:
                last_phones, last_results = analyze_frame(
                    frame, detector, pose_analyzer, behavior_engine
                )
                update_stats(stats, last_results)

            draw_frame(frame, last_phones, last_results)
            cv2.putText(
                frame,
                f"FPS {display_fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            fps_count += 1
            elapsed = time.time() - fps_started_at
            if elapsed >= 1:
                display_fps = fps_count / elapsed
                fps_count, fps_started_at = 0, time.time()
                render_stats(stats_placeholder, stats)
                status.success("Camera đang chạy qua mạng Wi-Fi 🔴")
    finally:
        reader.release()
        st.session_state.active_mode = None


def run_uploaded_video(
    uploaded_video,
    detector,
    pose_analyzer,
    process_every_n,
    max_width,
    frame_placeholder,
    stats_placeholder,
    status,
    progress_placeholder,
):
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    temp_path = None
    capture = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_video.getbuffer())
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            status.error("Không đọc được video. Hãy thử định dạng MP4, AVI, MOV hoặc MKV.")
            return

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        behavior_engine = BehaviorEngine()
        reset_tracker(pose_analyzer)
        stats = st.session_state.stats
        last_phones, last_results = [], []
        frame_index = 0
        started_at = time.time()
        completed = False

        while st.session_state.active_mode == "video":
            ok, frame = capture.read()
            if not ok:
                completed = True
                break

            frame = resize_frame(frame, max_width)
            frame_index += 1
            if (frame_index - 1) % process_every_n == 0:
                last_phones, last_results = analyze_frame(
                    frame, detector, pose_analyzer, behavior_engine
                )
                update_stats(stats, last_results)

            draw_frame(frame, last_phones, last_results)
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            if total_frames > 0:
                progress_placeholder.progress(min(frame_index / total_frames, 1.0))
            if frame_index % 10 == 0:
                render_stats(stats_placeholder, stats)
                status.info(f"Đang xử lý frame {frame_index}/{total_frames or '?'}…")

        render_stats(stats_placeholder, stats)
        if completed:
            elapsed = time.time() - started_at
            status.success(
                f"Đã xử lý xong {frame_index} frame trong {elapsed:.1f} giây."
            )
            progress_placeholder.progress(1.0)
        else:
            status.info("Đã dừng xử lý video.")
    finally:
        if capture is not None:
            capture.release()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        st.session_state.active_mode = None


if "active_mode" not in st.session_state:
    st.session_state.active_mode = None
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "models" not in st.session_state:
    st.session_state.models = load_models()

detector, pose_analyzer = st.session_state.models

with st.sidebar:
    st.header("Hiệu năng")
    process_every_n = st.slider(
        "Xử lý mỗi N frame",
        1,
        5,
        2,
        help="Số lớn hơn giúp chạy nhanh hơn nhưng có thể bỏ qua hành vi ngắn.",
    )
    max_width = st.select_slider(
        "Độ rộng xử lý tối đa (px)",
        options=[480, 640, 960, 1280],
        value=960,
        help="Giảm độ rộng nếu máy không có GPU.",
    )

live_panel, upload_panel = st.columns(2, gap="large")

with live_panel:
    st.subheader("📱 Camera không dây / Webcam")
    live_source_type = st.selectbox(
        "Chọn nguồn camera",
        ["Điện thoại qua Wi-Fi", "Webcam máy tính", "URL camera khác"],
    )

    if live_source_type == "Điện thoại qua Wi-Fi":
        source = st.text_input(
            "URL camera từ điện thoại",
            placeholder="http://192.168.1.20:8080/video",
            help="Điện thoại và máy tính phải cùng mạng Wi-Fi.",
        ).strip()
        st.caption(
            "Mở IP Webcam/DroidCam trên điện thoại, bật server rồi nhập URL "
            "mà ứng dụng hiển thị. Không cần ADB hoặc cáp USB."
        )
    elif live_source_type == "Webcam máy tính":
        source = int(st.number_input("Số camera", 0, 9, 0, step=1))
    else:
        source = st.text_input(
            "URL luồng camera",
            placeholder="http://camera.local:8080/video",
        ).strip()

    start_live = st.button(
        "▶️ Bắt đầu camera",
        type="primary",
        use_container_width=True,
    )

with upload_panel:
    st.subheader("🎞️ Phân tích video có sẵn")
    uploaded_video = st.file_uploader(
        "Chọn video",
        type=["mp4", "avi", "mov", "mkv", "m4v"],
        help="Video chỉ được lưu vào file tạm trong lúc phân tích.",
    )
    if uploaded_video is not None:
        size_mb = uploaded_video.size / (1024 * 1024)
        st.caption(f"{uploaded_video.name} — {size_mb:.1f} MB")
    start_video = st.button(
        "▶️ Phân tích video",
        type="primary",
        use_container_width=True,
    )

if start_live:
    if isinstance(source, str) and not source:
        st.error("Hãy nhập URL camera trước khi bắt đầu.")
    else:
        st.session_state.active_mode = "live"
        st.session_state.stats = {}

if start_video:
    if uploaded_video is None:
        st.error("Hãy chọn một file video trước khi phân tích.")
    else:
        st.session_state.active_mode = "video"
        st.session_state.stats = {}

stop_processing = st.button(
    "⏹️ Dừng xử lý",
    use_container_width=True,
    disabled=st.session_state.active_mode is None,
)

if stop_processing:
    st.session_state.active_mode = None

video_column, stats_column = st.columns([3, 2], gap="large")
with video_column:
    st.subheader("Kết quả nhận diện")
    status = st.empty()
    progress_placeholder = st.empty()
    frame_placeholder = st.empty()

with stats_column:
    st.subheader("Tổng kết theo từng ID")
    stats_placeholder = st.empty()

if st.session_state.active_mode == "live":
    run_live_source(
        source,
        detector,
        pose_analyzer,
        process_every_n,
        max_width,
        frame_placeholder,
        stats_placeholder,
        status,
    )
elif st.session_state.active_mode == "video" and uploaded_video is not None:
    run_uploaded_video(
        uploaded_video,
        detector,
        pose_analyzer,
        process_every_n,
        max_width,
        frame_placeholder,
        stats_placeholder,
        status,
        progress_placeholder,
    )

render_stats(stats_placeholder, st.session_state.stats)
