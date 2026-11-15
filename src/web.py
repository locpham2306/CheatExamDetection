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


st.set_page_config(
    page_title="ExamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #0f172a;
        --slate: #475569;
        --muted: #64748b;
        --line: #e2e8f0;
        --surface: #ffffff;
        --canvas: #f4f7fb;
        --indigo: #4f46e5;
        --teal: #0f9f8f;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 90% 0%, rgba(79, 70, 229, 0.08), transparent 24rem),
            var(--canvas);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #172554 100%);
        border-right: 0;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] [data-baseweb="slider"] * {
        color: inherit;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.12);
    }

    .nav-label {
        margin: 0 0 0.55rem;
        color: #64748b;
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] .stButton {
        margin-bottom: 0.38rem;
    }

    [data-testid="stSidebar"] .stButton > button {
        min-height: 44px;
        justify-content: flex-start;
        padding: 0.7rem 0.9rem;
        border: 1px solid transparent;
        border-radius: 11px;
        background: rgba(255, 255, 255, 0.035);
        color: #cbd5e1;
        box-shadow: none;
        font-size: 0.86rem;
        font-weight: 650;
        transition: border-color 140ms ease, background 140ms ease,
                    transform 140ms ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(148, 163, 184, 0.22);
        background: rgba(255, 255, 255, 0.075);
        color: white;
        transform: translateX(2px);
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        border-color: rgba(94, 234, 212, 0.28);
        background: linear-gradient(
            90deg,
            rgba(79, 70, 229, 0.34),
            rgba(20, 184, 166, 0.14)
        );
        color: white;
        box-shadow: inset 3px 0 0 #5eead4;
    }

    [data-testid="stSidebar"] .stButton > button svg {
        width: 20px;
        height: 20px;
        margin-right: 0.35rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 0.25rem 0 1.6rem;
    }

    .sidebar-logo {
        display: grid;
        width: 42px;
        height: 42px;
        place-items: center;
        border-radius: 13px;
        background: linear-gradient(135deg, #6366f1, #14b8a6);
        color: white;
        font-size: 1.15rem;
        font-weight: 800;
        box-shadow: 0 10px 25px rgba(20, 184, 166, 0.2);
    }

    .sidebar-brand strong {
        display: block;
        color: white;
        font-size: 1.05rem;
        letter-spacing: -0.02em;
    }

    .sidebar-brand span {
        color: #94a3b8;
        font-size: 0.75rem;
    }

    .sidebar-status {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 0.65rem 0 0.85rem;
        padding: 0.7rem 0.8rem;
        border: 1px solid rgba(45, 212, 191, 0.22);
        border-radius: 11px;
        background: rgba(20, 184, 166, 0.1);
        color: #ccfbf1;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .sidebar-status::before {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #2dd4bf;
        content: "";
        box-shadow: 0 0 0 4px rgba(45, 212, 191, 0.12);
    }

    .model-meta {
        display: grid;
        gap: 0.55rem;
        margin: 0.15rem 0 0.9rem;
    }

    .model-meta-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.6rem;
        color: #94a3b8;
        font-size: 0.72rem;
    }

    .model-meta-row strong {
        overflow: hidden;
        color: #cbd5e1;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.68rem;
        font-weight: 550;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.035) !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:focus {
        background: transparent !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
        fill: #94a3b8 !important;
        color: #94a3b8 !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] li {
        color: #cbd5e1 !important;
    }

    [data-testid="stSidebar"] code {
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.45) !important;
        color: #99f6e4 !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        margin-bottom: 1.35rem;
        padding: 1.8rem 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        background: linear-gradient(120deg, #0f172a 0%, #172554 62%, #134e4a 130%);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.14);
    }

    .hero::after {
        position: absolute;
        right: -5rem;
        bottom: -7rem;
        width: 20rem;
        height: 20rem;
        border: 1px solid rgba(45, 212, 191, 0.18);
        border-radius: 50%;
        content: "";
        box-shadow: 0 0 0 3rem rgba(45, 212, 191, 0.035),
                    0 0 0 6rem rgba(99, 102, 241, 0.025);
    }

    .hero-kicker {
        margin-bottom: 0.55rem;
        color: #5eead4;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        color: white;
        font-size: clamp(1.65rem, 3vw, 2.35rem);
        font-weight: 750;
        letter-spacing: -0.045em;
        line-height: 1.1;
    }

    .hero p {
        position: relative;
        z-index: 1;
        max-width: 760px;
        margin: 0.75rem 0 1.15rem;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .hero-tags {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }

    .hero-tag {
        padding: 0.35rem 0.7rem;
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.07);
        color: #e2e8f0;
        font-size: 0.72rem;
        font-weight: 650;
    }

    .section-label {
        margin: 0.35rem 0 0.85rem;
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .card-heading {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        margin-bottom: 0.35rem;
    }

    .card-icon {
        display: grid;
        flex: 0 0 auto;
        width: 40px;
        height: 40px;
        place-items: center;
        border-radius: 12px;
        background: #eef2ff;
        color: var(--indigo);
        font-size: 1.05rem;
    }

    .card-heading h3 {
        margin: 0;
        color: var(--navy);
        font-size: 1.02rem;
        letter-spacing: -0.02em;
    }

    .card-heading p {
        margin: 0.2rem 0 0;
        color: var(--muted);
        font-size: 0.78rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.94);
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.045);
    }

    .stButton > button {
        min-height: 2.8rem;
        border-radius: 11px;
        font-weight: 700;
        transition: transform 120ms ease, box-shadow 120ms ease;
    }

    .stButton > button[kind="primary"] {
        border: 0;
        background: linear-gradient(110deg, #4f46e5, #4338ca);
        box-shadow: 0 8px 18px rgba(79, 70, 229, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed #c7d2fe;
        border-radius: 12px;
        background: #f8faff;
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    .monitor-empty {
        display: grid;
        min-height: 390px;
        place-items: center;
        border: 1.5px dashed #cbd5e1;
        border-radius: 14px;
        background: linear-gradient(145deg, #f8fafc, #f1f5f9);
        text-align: center;
    }

    .monitor-empty-icon {
        display: grid;
        width: 58px;
        height: 58px;
        margin: 0 auto 0.9rem;
        place-items: center;
        border-radius: 18px;
        background: white;
        color: #6366f1;
        font-size: 1.45rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    }

    .monitor-empty strong {
        display: block;
        margin-bottom: 0.25rem;
        color: #334155;
        font-size: 0.95rem;
    }

    .monitor-empty span {
        color: #94a3b8;
        font-size: 0.8rem;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.65rem;
        margin-bottom: 1rem;
    }

    .summary-card {
        padding: 0.85rem;
        border: 1px solid #e2e8f0;
        border-radius: 13px;
        background: #f8fafc;
    }

    .summary-card span {
        display: block;
        margin-bottom: 0.25rem;
        color: #64748b;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .summary-card strong {
        color: #0f172a;
        font-size: 1.35rem;
        letter-spacing: -0.04em;
    }

    .summary-card.danger {
        border-color: #fecaca;
        background: #fff7f7;
    }

    .summary-card.danger strong {
        color: #dc2626;
    }

    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 12px;
    }

    @media (max-width: 800px) {
        .block-container { padding: 1rem; }
        .hero { padding: 1.4rem; border-radius: 17px; }
        .monitor-empty { min-height: 270px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">AI Proctoring Dashboard</div>
        <h1>ExamGuard AI</h1>
        <p>Giám sát phòng thi theo thời gian thực, phát hiện điện thoại và hành vi
        quay đầu từ camera hoặc video có sẵn.</p>
        <div class="hero-tags">
            <span class="hero-tag">● Model sẵn sàng</span>
            <span class="hero-tag">YOLO Phone Detection</span>
            <span class="hero-tag">Pose Tracking</span>
            <span class="hero-tag">Không cần cáp USB</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

RISK_COLORS = {
    "LOW": (0, 255, 0),
    "MEDIUM": (0, 165, 255),
    "HIGH": (0, 0, 255),
}


def load_models():
    # best_phone_detection.pt -> điện thoại | yolov8n-pose -> người + tracking + hướng đầu
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


def render_summary(placeholder, stats):
    students = len(stats)
    phone_events = sum(item["Frame có điện thoại"] for item in stats.values())
    head_events = sum(item["Frame quay đầu"] for item in stats.values())
    high_risk = sum(
        item["Điểm rủi ro cao nhất"] >= config.RISK_LEVEL_HIGH
        for item in stats.values()
    )
    placeholder.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <span>Đối tượng</span><strong>{students}</strong>
            </div>
            <div class="summary-card danger">
                <span>Nguy cơ cao</span><strong>{high_risk}</strong>
            </div>
            <div class="summary-card">
                <span>Frame có điện thoại</span><strong>{phone_events}</strong>
            </div>
            <div class="summary-card">
                <span>Frame quay đầu</span><strong>{head_events}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
    source_type,
    detector,
    pose_analyzer,
    process_every_n,
    max_width,
    frame_placeholder,
    summary_placeholder,
    stats_placeholder,
    status,
):
    reader = LatestFrameReader(source)
    if not reader.is_opened():
        reader.release()
        st.session_state.active_mode = None
        if source_type == "Webcam máy tính":
            status.error(
                f"Không mở được camera số {source}. Hãy đóng ứng dụng khác đang "
                "dùng camera hoặc thử số camera khác (DroidCam thường là 1)."
            )
        else:
            status.error(
                "Không mở được luồng camera. Hãy kiểm tra URL, kết nối mạng và "
                "đảm bảo ứng dụng trên điện thoại đang phát video."
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
                render_summary(summary_placeholder, stats)
                render_stats(stats_placeholder, stats)
                status.success("Đang giám sát trực tiếp • Kết nối ổn định")
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
    summary_placeholder,
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
                render_summary(summary_placeholder, stats)
                render_stats(stats_placeholder, stats)
                status.info(f"Đang xử lý frame {frame_index}/{total_frames or '?'}…")

        render_summary(summary_placeholder, stats)
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
    with st.spinner("Đang khởi tạo mô hình nhận diện…"):
        st.session_state.models = load_models()

detector, pose_analyzer = st.session_state.models


if "workspace_screen" not in st.session_state:
    st.session_state.workspace_screen = "Camera trực tiếp"


def select_workspace(workspace_name):
    if st.session_state.workspace_screen != workspace_name:
        st.session_state.workspace_screen = workspace_name
        st.session_state.active_mode = None
        st.session_state.stats = {}


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">EG</div>
            <div><strong>ExamGuard</strong><span>Control Center</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-label">Menu chính</div>', unsafe_allow_html=True)
    st.button(
        "Camera trực tiếp",
        key="nav_camera",
        icon=":material/videocam:",
        type=(
            "primary"
            if st.session_state.workspace_screen == "Camera trực tiếp"
            else "secondary"
        ),
        use_container_width=True,
        on_click=select_workspace,
        args=("Camera trực tiếp",),
    )
    st.button(
        "Phân tích video",
        key="nav_video",
        icon=":material/movie:",
        type=(
            "primary"
            if st.session_state.workspace_screen == "Phân tích video"
            else "secondary"
        ),
        use_container_width=True,
        on_click=select_workspace,
        args=("Phân tích video",),
    )
    workspace = st.session_state.workspace_screen
    st.divider()
    st.markdown("#### Cấu hình xử lý")
    st.caption("Cân bằng giữa tốc độ và độ chính xác nhận diện.")
    process_every_n = st.slider(
        "Tần suất phân tích",
        1,
        5,
        2,
        help="Phân tích mỗi N frame. Số nhỏ chính xác hơn nhưng dùng nhiều tài nguyên hơn.",
    )
    max_width = st.select_slider(
        "Độ phân giải xử lý",
        options=[480, 640, 960, 1280],
        value=960,
        help="Giảm độ rộng nếu máy không có GPU.",
    )
    st.divider()
    st.markdown("#### Trạng thái hệ thống")
    st.markdown(
        '<div class="sidebar-status">Model điện thoại đã sẵn sàng</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="model-meta">
            <div class="model-meta-row">
                <strong>best_phone_detection.pt</strong><span>class phone</span>
            </div>
            <div class="model-meta-row">
                <strong>yolov8n-pose.pt</strong><span>pose &amp; tracking</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    with st.expander("Hướng dẫn nhanh"):
        st.markdown(
            """
            - Webcam laptop thường là camera **0**.
            - DroidCam thường xuất hiện ở camera **1**.
            - IP Webcam cần URL kết thúc bằng `/video`.
            - Giảm độ phân giải nếu FPS thấp.
            """
        )

start_live = False
start_video = False
source = None
live_source_type = None
uploaded_video = None

if workspace == "Camera trực tiếp":
    st.markdown(
        '<div class="section-label">01 · Thiết lập camera trực tiếp</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-heading">
                <div class="card-icon">◉</div>
                <div><h3>Kết nối camera</h3>
                <p>Chọn webcam laptop, DroidCam hoặc camera phát qua mạng</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        camera_form, camera_guide = st.columns([2, 1], gap="large")
        with camera_form:
            live_source_type = st.selectbox(
                "Nguồn camera",
                ["Webcam máy tính", "Điện thoại qua Wi-Fi", "URL camera khác"],
            )

            if live_source_type == "Điện thoại qua Wi-Fi":
                source = st.text_input(
                    "URL camera từ điện thoại",
                    placeholder="http://192.168.1.20:8080/video",
                    help="Điện thoại và máy tính phải cùng mạng Wi-Fi.",
                ).strip()
                st.caption("Dành cho IP Webcam hoặc ứng dụng cung cấp URL video.")
            elif live_source_type == "Webcam máy tính":
                source = int(st.number_input("Số camera", 0, 9, 0, step=1))
                st.caption("Webcam laptop thường là 0; DroidCam thường là 1.")
            else:
                source = st.text_input(
                    "URL luồng camera",
                    placeholder="http://camera.local:8080/video",
                ).strip()
                st.caption("Hỗ trợ HTTP/MJPEG và các URL OpenCV có thể đọc.")

        with camera_guide:
            st.info(
                "**Kết nối nhanh**\n\n"
                "Camera laptop: chọn số **0**.  \n"
                "DroidCam/iPhone: mở Client rồi thử số **1**.  \n"
                "IP Webcam: nhập URL kết thúc bằng `/video`."
            )

        start_live = st.button(
            "Bắt đầu giám sát camera  →",
            type="primary",
            use_container_width=True,
        )
else:
    st.markdown(
        '<div class="section-label">01 · Tải video cần phân tích</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-heading">
                <div class="card-icon">▶</div>
                <div><h3>Phân tích video có sẵn</h3>
                <p>Tải bản ghi lên để quét điện thoại và hành vi quay đầu</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        upload_form, upload_guide = st.columns([2, 1], gap="large")
        with upload_form:
            uploaded_video = st.file_uploader(
                "Tệp video",
                type=["mp4", "avi", "mov", "mkv", "m4v"],
                help="Video chỉ được lưu vào file tạm trong lúc phân tích.",
            )
            if uploaded_video is not None:
                size_mb = uploaded_video.size / (1024 * 1024)
                st.caption(f"Đã chọn: {uploaded_video.name} · {size_mb:.1f} MB")
            else:
                st.caption("MP4, AVI, MOV, MKV hoặc M4V · tối đa 200 MB")
        with upload_guide:
            st.info(
                "**Cách hoạt động**\n\n"
                "Video được xử lý cục bộ và chỉ lưu tạm trong lúc phân tích. "
                "Kết quả sẽ cập nhật theo từng ID ở bảng bên dưới."
            )

        start_video = st.button(
            "Bắt đầu phân tích video  →",
            type="primary",
            use_container_width=True,
        )

if workspace == "Camera trực tiếp" and start_live:
    if isinstance(source, str) and not source:
        st.error("Hãy nhập URL camera trước khi bắt đầu.")
    else:
        st.session_state.active_mode = "live"
        st.session_state.stats = {}

if workspace == "Phân tích video" and start_video:
    if uploaded_video is None:
        st.error("Hãy chọn một file video trước khi phân tích.")
    else:
        st.session_state.active_mode = "video"
        st.session_state.stats = {}

st.markdown("<br>", unsafe_allow_html=True)
monitor_title, monitor_action = st.columns([5, 1])
with monitor_title:
    st.markdown(
        (
            '<div class="section-label">02 · Giám sát trực tiếp</div>'
            if workspace == "Camera trực tiếp"
            else '<div class="section-label">02 · Kết quả phân tích</div>'
        ),
        unsafe_allow_html=True,
    )
with monitor_action:
    stop_processing = st.button(
        "Dừng xử lý",
        use_container_width=True,
        disabled=st.session_state.active_mode is None,
    )

if stop_processing:
    st.session_state.active_mode = None

video_column, stats_column = st.columns([5, 3], gap="medium")
with video_column:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="card-heading">
                <div class="card-icon">⌾</div>
                <div><h3>{'Luồng camera nhận diện' if workspace == 'Camera trực tiếp' else 'Video đang phân tích'}</h3>
                <p>{'Theo dõi và đánh dấu hành vi theo thời gian thực' if workspace == 'Camera trực tiếp' else 'Xem tiến trình và khung hình đã được đánh dấu'}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        status = st.empty()
        progress_placeholder = st.empty()
        frame_placeholder = st.empty()
        if st.session_state.active_mode is None:
            frame_placeholder.markdown(
                """
                <div class="monitor-empty"><div>
                    <div class="monitor-empty-icon">⌾</div>
                    <strong>{empty_title}</strong>
                    <span>{empty_help}</span>
                </div></div>
                """.format(
                    empty_title=(
                        "Camera chưa được kết nối"
                        if workspace == "Camera trực tiếp"
                        else "Chưa có video phân tích"
                    ),
                    empty_help=(
                        "Chọn nguồn camera ở phía trên và bắt đầu giám sát"
                        if workspace == "Camera trực tiếp"
                        else "Tải video ở phía trên để bắt đầu phân tích"
                    ),
                ),
                unsafe_allow_html=True,
            )

with stats_column:
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-heading">
                <div class="card-icon">▥</div>
                <div><h3>Tổng quan cảnh báo</h3>
                <p>Số liệu được cập nhật trong quá trình xử lý</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        summary_placeholder = st.empty()
        stats_placeholder = st.empty()

if workspace == "Camera trực tiếp" and st.session_state.active_mode == "live":
    run_live_source(
        source,
        live_source_type,
        detector,
        pose_analyzer,
        process_every_n,
        max_width,
        frame_placeholder,
        summary_placeholder,
        stats_placeholder,
        status,
    )
elif (
    workspace == "Phân tích video"
    and st.session_state.active_mode == "video"
    and uploaded_video is not None
):
    run_uploaded_video(
        uploaded_video,
        detector,
        pose_analyzer,
        process_every_n,
        max_width,
        frame_placeholder,
        summary_placeholder,
        stats_placeholder,
        status,
        progress_placeholder,
    )

render_summary(summary_placeholder, st.session_state.stats)
render_stats(stats_placeholder, st.session_state.stats)
