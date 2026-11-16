import os
import subprocess
import tempfile
import threading
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
from behavior_engine import BehaviorEngine
from detector import ObjectDetector
from pose_analyzer import PoseAnalyzer
from seat_assigner import SeatIDAssigner


st.set_page_config(page_title="Exam Cheat Detection", layout="wide")
st.title("Phát hiện gian lận bằng camera ")

RISK_COLORS = {"LOW": (0, 255, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}


# ----------------------------------------------------------------------------
# Model / tiện ích
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    # best_phone_detection.pt -> điện thoại | yolov8s-pose -> người + hướng đầu
    return ObjectDetector(), PoseAnalyzer()


def reset_tracker(pose_analyzer):
    # Chỉ cần khi dùng tracker của YOLO (config.USE_TRACKER = True)
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
    """Camera trực tiếp: đọc ở luồng riêng và luôn giữ frame mới nhất."""

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


class VideoFileReader:
    """File video: đọc tuần tự từng frame, không bỏ sót frame nào."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.failed = False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def is_opened(self):
        return self.cap.isOpened()

    def start(self):
        return self

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self.failed = True
            return None
        return frame

    def release(self):
        self.cap.release()


# ----------------------------------------------------------------------------
# Vẽ
# ----------------------------------------------------------------------------
def put_text(frame, text, x, y, color, scale=0.5, thick=1):
    """Chữ có viền đen để dễ đọc trên nền phức tạp."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_frame(frame, phones, results, label_fn):
    for phone in phones:
        x1, y1, x2, y2 = map(int, phone["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        put_text(frame, f"PHONE {phone['conf']:.2f}", x1, max(20, y1 - 8), (0, 0, 255), 0.55, 2)

    for r in results:
        x1, y1, x2, y2 = map(int, r["box"])
        color = RISK_COLORS[r["risk_level"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Nhãn ngắn gọn: ghế + điểm rủi ro (+ hướng đầu nếu đang quay)
        label = f"{label_fn(r['track_id'])} {r['risk_score']}%"
        if r["head_dir"] != "CENTER":
            label += f" {r['head_dir']}"
        put_text(frame, label, x1, max(20, y1 - 8), color, 0.5, 1)
        if r["risk_level"] != "LOW" and r["rules"]:
            put_text(frame, " | ".join(r["rules"]), x1, min(frame.shape[0] - 8, y2 + 18), color, 0.45, 1)
    return frame


def draw_seat_map(frame, seat_ids):
    """Vẽ số ghế lớn lên khung hình để người dùng đối chiếu khi gán hàng/cột."""
    out = frame.copy()
    for sid, s in seat_ids.seats.items():
        x, y = s["pos"]
        w, h = s["w"], s["h"]
        cv2.rectangle(out, (int(x - w / 2), int(y)), (int(x + w / 2), int(y + h)), (0, 255, 255), 2)
        scale = max(0.7, w / 70)
        text = str(sid)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        put_text(out, text, int(x - tw / 2), int(y + h * 0.3 + th / 2), (0, 255, 255), scale, 2)
    return out


# ----------------------------------------------------------------------------
# Thống kê / cảnh báo
# ----------------------------------------------------------------------------
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


def render_stats(placeholder, stats, seat_ids=None):
    if not stats:
        placeholder.info("Chưa có dữ liệu.")
        return
    df = pd.DataFrame.from_dict(stats, orient="index")
    if seat_ids is not None:
        if seat_ids.has_layout:
            pos = {sid: seat_ids.position(sid) or (None, None) for sid in df.index}
            df.insert(0, "Cột", [pos[i][1] for i in df.index])
            df.insert(0, "Hàng", [pos[i][0] for i in df.index])
        df.index = [seat_ids.label(i) for i in df.index]
        df.index.name = "Ghế"
    else:
        df.index.name = "ID"
    df = df.sort_values("Điểm rủi ro cao nhất", ascending=False)

    def highlight_high_risk(row):
        if row["Điểm rủi ro cao nhất"] >= config.RISK_LEVEL_HIGH:
            return ["background-color: #ff4b4b; color: white; font-weight: bold"] * len(row)
        return [""] * len(row)

    placeholder.dataframe(df.style.apply(highlight_high_risk, axis=1),
                          use_container_width=True)


def _seat_sort_key(seat_ids, sid):
    pos = seat_ids.position(sid) if seat_ids else None
    return (pos or (999, 999), int(sid))


def render_alerts(placeholder, results, seat_ids):
    """Thí sinh đang ở mức nguy cơ cao ở thời điểm hiện tại."""
    high = [r for r in results if r["risk_level"] == "HIGH"]
    if not high:
        placeholder.success("Hiện không có thí sinh nguy cơ cao.")
        return
    lines = []
    for r in sorted(high, key=lambda r: _seat_sort_key(seat_ids, r["track_id"])):
        where = seat_ids.long_label(r["track_id"]) if seat_ids else f"ID {r['track_id']}"
        why = "; ".join(r["rules"]) if r["rules"] else "-"
        lines.append(f"🚨 **{where}** — rủi ro {r['risk_score']}% — {why}")
    placeholder.error("\n\n".join(lines))


def render_high_summary(placeholder, stats, seat_ids):
    """Tổng kết: các thí sinh từng đạt mức nguy cơ cao trong cả phiên."""
    high = [(sid, s) for sid, s in stats.items()
            if s["Điểm rủi ro cao nhất"] >= config.RISK_LEVEL_HIGH]
    if not high:
        placeholder.success("Không có thí sinh nào đạt mức nguy cơ cao.")
        return
    high.sort(key=lambda t: _seat_sort_key(seat_ids, t[0]))
    lines = []
    for sid, s in high:
        where = seat_ids.long_label(sid) if seat_ids else f"ID {sid}"
        lines.append(f"🚨 **{where}** — rủi ro cao nhất {s['Điểm rủi ro cao nhất']}% "
                     f"— {s['Frame nguy cơ cao']} frame nguy cơ cao")
    placeholder.error("Thí sinh từng đạt mức nguy cơ cao:\n\n" + "\n\n".join(lines))


# ----------------------------------------------------------------------------
# Trạng thái phiên / giao diện dùng chung
# ----------------------------------------------------------------------------
def get_state(key):
    if key not in st.session_state:
        st.session_state[key] = {
            "running": False, "stats": {}, "seat_ids": None,
            "video_path": None, "video_sig": None,
            "calib": None, "layout": None, "calib_ver": 0,
        }
    return st.session_state[key]


def build_panel():
    """Khu vực hiển thị (video + thống kê) của một tab."""
    col_video, col_stats = st.columns([3, 2], gap="large")
    ui = {}
    with col_video:
        st.subheader("🎥 Video giám sát")
        ui["status"] = st.empty()
        ui["progress"] = st.empty()
        ui["seat_info"] = st.empty()
        ui["frame"] = st.empty()
    with col_stats:
        st.subheader("🚨 Nguy cơ cao")
        ui["alert"] = st.empty()
        st.subheader("📊 Tổng kết theo từng chỗ ngồi")
        ui["stats"] = st.empty()
    return ui


def layout_editor(key):
    """Bước ②: người dùng đối chiếu số ghế trên ảnh và nhập Hàng/Cột thật."""
    state = get_state(key)
    calib = state["calib"]
    expanded = calib is not None and state["layout"] is None
    with st.expander("② Gán Hàng - Cột cho từng chỗ ngồi", expanded=expanded):
        if calib is None:
            st.info("Chưa hiệu chỉnh. Khi mọi người đã ngồi yên, bấm **① Hiệu chỉnh chỗ ngồi**. "
                    "Nếu bỏ qua bước này, ghế chỉ hiển thị dạng #số, không có Hàng/Cột.")
            return

        c1, c2 = st.columns([3, 2], gap="large")
        c1.image(calib["snapshot"], channels="RGB", use_container_width=True,
                 caption=f"Số màu vàng = mã ghế. Đã nhận diện {len(calib['seats'])} chỗ.")

        ids = sorted(int(i) for i in calib["seats"])
        base = state["layout"] or calib["suggest"]
        df = pd.DataFrame({
            "Ghế": ids,
            "Hàng": [base.get(i, (None, None))[0] for i in ids],
            "Cột": [base.get(i, (None, None))[1] for i in ids],
        })
        with c2:
            st.caption("Điền Hàng/Cột thật theo sơ đồ lớp (bảng đã điền sẵn gợi ý tự động, "
                       "có thể sai nếu camera đặt chéo góc). Bấm Lưu khi xong.")
            with st.form(f"layout_form_{key}_{state['calib_ver']}"):
                edited = st.data_editor(
                    df, hide_index=True, disabled=["Ghế"], use_container_width=True,
                    key=f"layout_editor_{key}_{state['calib_ver']}", height=360,
                    column_config={
                        "Hàng": st.column_config.NumberColumn("Hàng", min_value=1, step=1, format="%d"),
                        "Cột": st.column_config.NumberColumn("Cột", min_value=1, step=1, format="%d"),
                    })
                submitted = st.form_submit_button("💾 Lưu sơ đồ", type="primary")

            if submitted:
                layout = {}
                for _, r in edited.iterrows():
                    if pd.notna(r["Hàng"]) and pd.notna(r["Cột"]):
                        layout[int(r["Ghế"])] = (int(r["Hàng"]), int(r["Cột"]))
                state["layout"] = layout or None
                dup = pd.Series(list(layout.values())).value_counts()
                dup = [f"H{a}-C{b}" for (a, b), n in dup.items() if n > 1]
                if dup:
                    st.warning("Có nhiều ghế trùng vị trí: " + ", ".join(dup))
                st.success(f"Đã lưu sơ đồ cho {len(layout)}/{len(ids)} ghế.")
            elif state["layout"]:
                st.success(f"Đang dùng sơ đồ đã lưu ({len(state['layout'])} ghế).")

            if st.button("🗑️ Xoá hiệu chỉnh", key=f"calib_clear_{key}"):
                state["calib"] = None
                state["layout"] = None
                st.rerun()


# ----------------------------------------------------------------------------
# Bước ①: hiệu chỉnh chỗ ngồi + hướng đầu "bình thường" của từng ghế
# ----------------------------------------------------------------------------
def run_calibration(key, reader, is_file, ui, cfg, pose_analyzer):
    state = get_state(key)
    status = ui["status"]

    if not reader.is_opened():
        reader.release()
        status.error("Không mở được nguồn video/camera.")
        return
    reader.start()

    seat_ids = SeatIDAssigner(max_seats=cfg["max_seats"], confirm_frames=cfg["confirm_frames"],
                              match_ratio=0.6, row_ratio=cfg["row_ratio"],
                              row_from_bottom=cfg["row_from_bottom"])
    offsets = defaultdict(list)
    reset_tracker(pose_analyzer)

    n = cfg["process_every_n"]
    src_fps = getattr(reader, "fps", 25.0)
    secs = cfg["calib_seconds"]
    frame_idx = 0
    run_start = last_frame_time = time.time()
    last_frame = None
    status.info("Đang hiệu chỉnh chỗ ngồi... hãy để mọi người ngồi yên, nhìn như bình thường.")

    try:
        while True:
            frame = reader.read()
            if frame is None:
                if reader.failed or time.time() - last_frame_time > 10:
                    break
                time.sleep(0.005)
                continue
            last_frame_time = time.time()
            frame_idx += 1
            if is_file and frame_idx % n != 0:
                continue

            h, w = frame.shape[:2]
            if w > cfg["max_width"]:
                frame = cv2.resize(frame, (cfg["max_width"], int(h * cfg["max_width"] / w)))

            if frame_idx % n == 0:
                persons = seat_ids.update(pose_analyzer.analyze_pose(frame))
                for p in persons:
                    if p["track_id"] is not None and p.get("head_offset") is not None:
                        offsets[int(p["track_id"])].append(p["head_offset"])

            elapsed = (frame_idx / src_fps) if is_file else (time.time() - run_start)
            last_frame = frame
            ui["frame"].image(draw_seat_map(frame, seat_ids), channels="BGR", use_container_width=True)
            ui["progress"].progress(min(elapsed / secs, 1.0),
                                    text=f"Hiệu chỉnh {min(elapsed, secs):.0f}/{secs}s "
                                         f"— đã nhận diện {len(seat_ids.seats)} chỗ")
            if elapsed >= secs:
                break
    finally:
        reader.release()
    ui["progress"].empty()

    if not seat_ids.seats or last_frame is None:
        status.error("Không nhận diện được chỗ ngồi nào. Kiểm tra camera/video rồi thử lại.")
        return

    seat_ids.lock()
    baseline = {sid: float(np.median(v)) for sid, v in offsets.items() if len(v) >= 3}
    snapshot = cv2.cvtColor(draw_seat_map(last_frame, seat_ids), cv2.COLOR_BGR2RGB)
    state.update(
        calib={"seats": seat_ids.export_seats(), "baseline": baseline,
               "snapshot": snapshot, "suggest": seat_ids.suggest_layout()},
        layout=None,
        calib_ver=state["calib_ver"] + 1,
        stats={}, seat_ids=None,
    )
    st.rerun()


# ----------------------------------------------------------------------------
# Bước ③: giám sát (dùng chung cho ADB và video)
# ----------------------------------------------------------------------------
def run_monitor(key, reader, is_file, ui, cfg, detector, pose_analyzer):
    state = get_state(key)
    status = ui["status"]

    if not reader.is_opened():
        reader.release()
        state["running"] = False
        if is_file:
            status.error("Không mở được file video. Hãy thử file khác hoặc định dạng mp4 (H.264).")
        else:
            status.error(
                "Không mở được luồng camera. Hãy kiểm tra: app trên điện thoại đã bấm "
                "'Start server' chưa, cáp USB, USB debugging, và cổng/đường dẫn stream."
            )
        return
    reader.start()

    behavior_engine = BehaviorEngine()
    seat_ids = SeatIDAssigner(max_seats=cfg["max_seats"], confirm_frames=cfg["confirm_frames"],
                              match_ratio=0.6, row_ratio=cfg["row_ratio"],
                              row_from_bottom=cfg["row_from_bottom"])
    baseline = {}
    calib = state.get("calib")
    if calib:
        seat_ids.load_seats(calib["seats"])             # dùng ghế đã hiệu chỉnh, không tạo ghế mới
        baseline = calib["baseline"]
        if state.get("layout"):
            seat_ids.set_manual_layout(state["layout"])
    state["seat_ids"] = seat_ids
    reset_tracker(pose_analyzer)

    stats = state["stats"]
    last_phones, last_results = [], []
    frame_idx = 0
    fps_count, fps_time, fps = 0, time.time(), 0.0
    run_start = last_frame_time = time.time()
    n = cfg["process_every_n"]
    src_fps = getattr(reader, "fps", 25.0)
    total = getattr(reader, "total_frames", 0)

    try:
        while True:
            frame = reader.read()
            if frame is None:
                if reader.failed:
                    if is_file:
                        status.success("✅ Đã xử lý xong video.")
                    else:
                        status.error("Mất kết nối với camera điện thoại.")
                    break
                if time.time() - last_frame_time > 10:
                    status.error("Mất kết nối với camera điện thoại.")
                    break
                time.sleep(0.005)
                continue
            last_frame_time = time.time()
            frame_idx += 1

            if is_file and frame_idx % n != 0:
                continue

            h, w = frame.shape[:2]
            if w > cfg["max_width"]:
                frame = cv2.resize(frame, (cfg["max_width"], int(h * cfg["max_width"] / w)))

            if frame_idx % n == 0:
                last_phones = detector.detect_phones(frame)
                persons = seat_ids.update(pose_analyzer.analyze_pose(frame))
                # Hướng đầu tính so với mức "bình thường" của từng ghế đã đo lúc hiệu chỉnh
                for p in persons:
                    if p["track_id"] is not None:
                        p["head_direction"] = PoseAnalyzer.direction_from_offset(
                            p.get("head_offset"), baseline.get(int(p["track_id"]), 0.0))
                last_results = behavior_engine.process_frame_data(persons, last_phones)
                update_stats(stats, last_results)

            elapsed = (frame_idx / src_fps) if is_file else (time.time() - run_start)
            if cfg["lock_after"] and not seat_ids.locked and elapsed >= cfg["lock_after"]:
                seat_ids.lock()

            frame = draw_frame(frame, last_phones, last_results, seat_ids.label)
            put_text(frame, f"FPS {fps:.1f}", 10, 25, (255, 255, 255), 0.7, 2)
            ui["frame"].image(frame, channels="BGR", use_container_width=True)

            if is_file and total > 0:
                ui["progress"].progress(min(frame_idx / total, 1.0), text=f"Frame {frame_idx}/{total}")

            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_count / (time.time() - fps_time)
                fps_count, fps_time = 0, time.time()
                render_stats(ui["stats"], stats, seat_ids)
                render_alerts(ui["alert"], last_results, seat_ids)
                status.success("Đang xử lý video 🎞️" if is_file else "Đang chạy trực tiếp 🔴")
                info = f"Chỗ ngồi: {len(seat_ids.seats)}"
                if not calib:
                    info += f"/{cfg['max_seats']} ({'🔒 đã chốt' if seat_ids.locked else '⏳ đang hiệu chỉnh'})"
                if seat_ids.has_layout:
                    rows, cols = seat_ids.grid_size()
                    info += f" | sơ đồ {rows} hàng x {cols} cột"
                ui["seat_info"].caption(info + f" | Phát hiện trong frame: {len(last_results)}")
    finally:
        reader.release()
        state["running"] = False

    render_stats(ui["stats"], stats, seat_ids)
    if is_file:
        render_high_summary(ui["alert"], stats, seat_ids)


# ----------------------------------------------------------------------------
# Giao diện
# ----------------------------------------------------------------------------
detector, pose_analyzer = load_models()
adb_state = get_state("adb")
video_state = get_state("video")

with st.sidebar:
    st.header("Hiệu năng")
    process_every_n = st.slider("Xử lý mỗi N frame", 1, 5, 2,
                                help="Số lớn hơn = chạy nhanh hơn nhưng bỏ qua một số frame.")
    max_width = st.select_slider("Độ rộng tối đa khi xử lý (px)",
                                 options=[480, 640, 960, 1280, 1920], value=1280,
                                 help="Giữ cao (>=1280) để bắt được người ngồi hàng cuối.")

    st.header("Hiệu chỉnh chỗ ngồi")
    max_seats = st.number_input("Số chỗ ngồi / thí sinh", 1, 200, 24,
                                help="ID tối đa sẽ không vượt quá số này.")
    calib_seconds = st.slider("Thời gian hiệu chỉnh (giây)", 5, 30, 10,
                              help="Mọi người ngồi yên, nhìn như bình thường. Hệ thống ghi nhận "
                                   "chỗ ngồi và hướng đầu 'bình thường' của từng người.")
    confirm_frames = st.slider("Số lần thấy để xác nhận 1 chỗ", 3, 30, 10)
    lock_after = st.slider("Tự chốt ghế nếu bỏ qua hiệu chỉnh (giây, 0 = không)", 0, 60, 15,
                           help="Chỉ dùng khi bấm Giám sát mà chưa hiệu chỉnh trước.")

    with st.expander("Gợi ý Hàng - Cột tự động"):
        row_ratio = st.slider("Độ lệch tối đa trong 1 hàng", 0.1, 1.0, 0.3, 0.05,
                              help="Chỉ để điền sẵn bảng ở bước ②. Bạn vẫn nên kiểm tra lại.")
        row_from_bottom = st.checkbox("Hàng 1 là hàng gần camera nhất", value=True)

cfg = {
    "process_every_n": process_every_n,
    "max_width": max_width,
    "max_seats": int(max_seats),
    "confirm_frames": int(confirm_frames),
    "calib_seconds": int(calib_seconds),
    "lock_after": int(lock_after),
    "row_ratio": float(row_ratio),
    "row_from_bottom": row_from_bottom,
}

tab_adb, tab_video = st.tabs(["📱 Camera điện thoại (ADB)", "🎞️ Tải video lên"])

# ---- Tab 1: ADB ----
with tab_adb:
    st.subheader("Camera điện thoại qua cáp USB (ADB)")
    c1, c2 = st.columns(2)
    port = c1.number_input("Cổng (port)", 1024, 65535, 8080,
                           help="IP Webcam dùng 8080, DroidCam dùng 4747.")
    path = c2.text_input("Đường dẫn stream", "/video")
    adb_source = f"http://127.0.0.1:{int(port)}{path}"
    st.code(adb_source)

    if st.button("🔍 Kiểm tra thiết bị ADB"):
        ok, msg = run_adb(["devices"])
        (st.success if ok else st.error)(msg)

    b1, b2, b3 = st.columns(3)
    adb_calib = b1.button("① Hiệu chỉnh chỗ ngồi", use_container_width=True, key="adb_calib")
    adb_start = b2.button("③ ▶️ Bắt đầu giám sát", type="primary", use_container_width=True, key="adb_start")
    adb_stop = b3.button("⏹️ Dừng", use_container_width=True, key="adb_stop")
    layout_editor("adb")
    adb_ui = build_panel()

# ---- Tab 2: Video tải lên ----
with tab_video:
    st.subheader("Phân tích video đã quay")
    uploaded = st.file_uploader("Chọn file video", type=["mp4", "avi", "mov", "mkv"],
                                key="video_uploader")
    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if video_state["video_sig"] != sig:
            old = video_state["video_path"]
            if old and os.path.exists(old):
                os.remove(old)
            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.getbuffer())
            tmp.close()
            video_state.update(video_path=tmp.name, video_sig=sig,
                               calib=None, layout=None, stats={}, seat_ids=None)  # video mới -> hiệu chỉnh lại
        st.caption(f"Đã tải lên: {uploaded.name} ({uploaded.size / 1_048_576:.1f} MB)")

    v1, v2, v3 = st.columns(3)
    video_calib = v1.button("① Hiệu chỉnh chỗ ngồi", use_container_width=True,
                            key="video_calib", disabled=uploaded is None)
    video_start = v2.button("③ ▶️ Phân tích video", type="primary", use_container_width=True,
                            key="video_start", disabled=uploaded is None)
    video_stop = v3.button("⏹️ Dừng", use_container_width=True, key="video_stop")
    layout_editor("video")
    video_ui = build_panel()

# ---- Xử lý nút bấm (làm hết trước khi chạy vòng lặp) ----
if adb_calib or adb_start:
    video_state["running"] = False
if video_calib or video_start:
    adb_state["running"] = False
if adb_start:
    adb_state.update(running=True, stats={}, seat_ids=None)
if video_start:
    video_state.update(running=True, stats={}, seat_ids=None)
if adb_stop:
    adb_state["running"] = False
if video_stop:
    video_state["running"] = False
if adb_calib:
    adb_state["running"] = False
if video_calib:
    video_state["running"] = False

render_stats(adb_ui["stats"], adb_state["stats"], adb_state["seat_ids"])
render_stats(video_ui["stats"], video_state["stats"], video_state["seat_ids"])


def video_path_ok():
    p = video_state["video_path"]
    return bool(p) and os.path.exists(p)


# ---- Chạy ----
if adb_calib or adb_state["running"]:
    ok, msg = run_adb(["forward", f"tcp:{int(port)}", f"tcp:{int(port)}"])
    if not ok:
        adb_state["running"] = False
        adb_ui["status"].error(f"adb forward thất bại: {msg}")
    elif adb_calib:
        run_calibration("adb", LatestFrameReader(adb_source), False, adb_ui, cfg, pose_analyzer)
    else:
        run_monitor("adb", LatestFrameReader(adb_source), False, adb_ui, cfg, detector, pose_analyzer)

elif video_calib or video_state["running"]:
    if not video_path_ok():
        video_state["running"] = False
        video_ui["status"].error("Chưa có video. Hãy tải file lên trước.")
    elif video_calib:
        run_calibration("video", VideoFileReader(video_state["video_path"]), True, video_ui, cfg, pose_analyzer)
    else:
        run_monitor("video", VideoFileReader(video_state["video_path"]), True, video_ui, cfg,
                    detector, pose_analyzer)