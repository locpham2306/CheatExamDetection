import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import time

from detector import ObjectDetector
from pose_analyzer import PoseAnalyzer
from behavior_engine import BehaviorEngine

# Cấu hình Trang Dashboard
st.set_page_config(page_title="Exam Cheat Detection MVP", layout="wide")

st.title("🚨 Hệ Thống Phát Hiện Gian Lận Trong Phòng Thi (Demo MVP)")
st.markdown("---")

# Khởi tạo các Module (Caching để tối ưu hiệu năng)
@st.cache_resource
def load_modules():
    detector = ObjectDetector()
    pose_analyzer = PoseAnalyzer()
    return detector, pose_analyzer

detector, pose_analyzer = load_modules()
behavior_engine = BehaviorEngine()

# Sidebar: Controls
st.sidebar.header("⚙️ Cấu Hình Input")
uploaded_file = st.sidebar.file_uploader("Tải lên Video/Ảnh phòng thi", type=["mp4", "avi", "mov", "jpg", "png"])

run_demo = st.sidebar.button("Chạy Kiểm Tra")

# Layout chính
col_video, col_logs = st.columns([2, 1])

with col_video:
    st.subheader("📹 Luồng Video Inference")
    st_frame = st.empty()

with col_logs:
    st.subheader("⚠️ Event Log & Risk Alerts")
    log_table_placeholder = st.empty()

event_logs = []

if run_demo and uploaded_file is not None:
    # Lưu file tạm để OpenCV đọc
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
        # Pipeline Inference (Giai đoạn B)
        persons, phones = detector.detect_and_track(frame)
        poses = pose_analyzer.analyze_pose(frame)
        analysis_results = behavior_engine.process_frame_data(persons, phones, poses)

        # Draw Overlay Bounding Boxes & Annotations
        for res in analysis_results:
            x1, y1, x2, y2 = map(int, res["box"])
            track_id = res["track_id"]
            risk_score = res["risk_score"]
            risk_level = res["risk_level"]

            # Set color according to risk
            color = (0, 255, 0) # Green LOW
            if risk_level == "MEDIUM":
                color = (0, 165, 255) # Orange
            elif risk_level == "HIGH":
                color = (0, 0, 255) # Red

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID #{track_id} | Risk: {risk_score}% ({res['head_dir']})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Record Log for Suspicious Events
            if risk_level in ["MEDIUM", "HIGH"]:
                timestamp = f"{int((frame_idx/fps)//60):02d}:{int((frame_idx/fps)%60):02d}"
                event_logs.append({
                    "Timestamp": timestamp,
                    "Person ID": f"Person #{track_id}",
                    "Risk Level": risk_level,
                    "Risk Score": f"{risk_score}%",
                    "Dấu hiệu": ", ".join(res["rules"])
                })

        # Render Frame on Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update Event Log UI
        if event_logs:
            df_logs = pd.DataFrame(event_logs).drop_duplicates(subset=["Timestamp", "Person ID", "Risk Level"]).tail(10)
            log_table_placeholder.dataframe(df_logs, use_container_width=True)

        time.sleep(0.01)

    cap.release()
    st.success("Đã hoàn tất xử lý Video!")
