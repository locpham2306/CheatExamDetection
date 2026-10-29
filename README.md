# Exam Cheat Detection

Demo phát hiện hành vi đáng ngờ trong phòng thi bằng camera. Hệ thống dùng model
custom `best.pt` để phát hiện điện thoại và `yolov8n-pose.pt` để nhận diện,
tracking người và ước lượng hướng đầu.

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy

WebRTC dùng camera trình duyệt:

```powershell
streamlit run src/app.py
```

Web giám sát camera URL, camera local hoặc điện thoại qua ADB:

```powershell
streamlit run src/web.py
```

Demo OpenCV với camera local:

```powershell
python src/live_demo.py
```

Nhấn `Q` hoặc `Esc` để đóng demo OpenCV.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Các model được đặt trong thư mục `model/`. `best.pt` phải có class `0` là
`phone`; nếu thay bằng model COCO thì cần cập nhật `PHONE_CLASS_ID` trong
`src/config.py`.
