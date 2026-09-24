# Exam Cheat Detection System (Demo MVP)

Hệ thống phát hiện gian lận phòng thi qua camera sử dụng Computer Vision & Deep Learning.

## Cấu trúc dự án
- `config.py`: Cấu hình thresholds, weights và đường dẫn model.
- `detector.py`: Nhận diện người & điện thoại, tích hợp ByteTrack tracking.
- `pose_analyzer.py`: Phân tích keypoints tư thế & hướng quay đầu (Head pose estimation).
- `behavior_engine.py`: Động cơ luật, liên kết object, phân tích thời gian (Temporal analysis) & tính Risk Score.
- `app.py`: Giao diện Web App Streamlit hiển thị real-time video và Event Log.

## Hướng dẫn cài đặt & chạy
1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```

2. Khởi chạy ứng dụng Streamlit:
   ```bash
   streamlit run app.py
   ```

3. Chạy camera trực tiếp, không dùng web:
   ```bash
   python live_demo.py
   ```

   Nhấn `Q` hoặc `Esc` để thoát.
