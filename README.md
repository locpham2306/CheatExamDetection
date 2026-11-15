# Exam Cheat Detection

Demo phát hiện hành vi đáng ngờ trong phòng thi bằng camera. Hệ thống dùng
`best_phone_detection.pt` (class `0`, `phone`) để phát hiện điện thoại và
`yolov8n-pose.pt` để nhận diện, tracking người và ước lượng hướng đầu.

## Cài đặt

Trên Windows, có thể cài tự động một lần bằng:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Hoặc cài thủ công:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy

Sau khi cài đặt, mở `src/app.py` trong VS Code và nhấn nút **Run Python
File**. Hoặc chạy lệnh:

```powershell
python src/app.py
```

Lệnh trên tự mở giao diện giám sát camera điện thoại qua Wi-Fi, webcam máy tính
hoặc video tải lên. Cũng có thể khởi chạy Streamlit trực tiếp:

```powershell
streamlit run src/web.py
```

Giao diện WebRTC cũ vẫn có thể chạy riêng bằng:

```powershell
streamlit run src/webrtc_app.py
```

### Camera điện thoại không dây

Với ứng dụng cung cấp URL video (ví dụ IP Webcam trên Android):

1. Kết nối điện thoại và máy tính vào cùng một mạng Wi-Fi.
2. Bật máy chủ video trong ứng dụng.
3. Chọn **Điện thoại qua Wi-Fi** và nhập URL đầy đủ, ví dụ
   `http://192.168.1.20:8080/video`.

Với DroidCam trên iPhone:

1. Kết nối iPhone và laptop cùng Wi-Fi, hoặc kết nối laptop vào Điểm truy cập cá
   nhân của iPhone.
2. Kết nối iPhone trong DroidCam Client trên Windows và kiểm tra Client đã hiện
   hình camera.
3. Trong ứng dụng này, chọn **Webcam máy tính** và thử camera số `1` (camera số
   `0` thường là webcam tích hợp của laptop). Không nhập URL DroidCam vào ô URL.

Không cần ADB, USB debugging hoặc cáp USB.

### Phân tích video

Trong cùng giao diện `src/web.py`, chọn video MP4, AVI, MOV, MKV hoặc M4V ở
khung **Phân tích video có sẵn**, sau đó nhấn **Phân tích video**.

Demo OpenCV với camera local:

```powershell
python src/live_demo.py
```

Nhấn `Q` hoặc `Esc` để đóng demo OpenCV.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Các model được đặt trong thư mục `model/`. Khi thay model phát hiện điện thoại,
cần cập nhật `YOLO_DETECTION_MODEL` và `PHONE_CLASS_ID` trong `src/config.py`
cho khớp tên file và class ID của model đó.
