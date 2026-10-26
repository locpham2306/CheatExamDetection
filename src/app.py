import threading

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from behavior_engine import BehaviorEngine
from detector import ObjectDetector
from pose_analyzer import PoseAnalyzer


st.set_page_config(page_title="Exam Cheat Detection", layout="wide")
st.title("🚨 Phát hiện gian lận thi trực tiếp")
st.caption("Cho phép truy cập camera, sau đó hệ thống sẽ nhận diện người, điện thoại và hướng quay đầu theo thời gian thực.")


@st.cache_resource
def load_models():
    return ObjectDetector(), PoseAnalyzer()


detector, pose_analyzer = load_models()


class ExamVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.behavior_engine = BehaviorEngine()
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        with self.lock:
            persons, phones = detector.detect_and_track(image)
            poses = pose_analyzer.analyze_pose(image)
            results = self.behavior_engine.process_frame_data(persons, phones, poses)

        for result in results:
            x1, y1, x2, y2 = map(int, result["box"])
            level = result["risk_level"]
            color = {
                "LOW": (0, 255, 0),
                "MEDIUM": (0, 165, 255),
                "HIGH": (0, 0, 255),
            }[level]

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = (
                f"ID #{result['track_id']} | Risk {result['risk_score']}% "
                f"| {result['head_dir']}"
            )
            cv2.putText(
                image,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

            if result["rules"]:
                warning = " | ".join(result["rules"])
                cv2.putText(
                    image,
                    warning,
                    (x1, min(image.shape[0] - 10, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="exam-live-camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    video_processor_factory=ExamVideoProcessor,
    desired_playing_state=True,
    async_processing=True,
)

st.info("Nếu camera chưa hiện, nhấn **START** và chọn **Allow/Cho phép** khi trình duyệt hỏi quyền camera.")
