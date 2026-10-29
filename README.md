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

Giao diện giám sát camera điện thoại qua Wi-Fi, webcam máy tính hoặc video tải
lên:

```powershell
streamlit run src/web.py
```

### Camera điện thoại không dây

1. Kết nối điện thoại và máy tính vào cùng một mạng Wi-Fi.
2. Mở ứng dụng phát camera như IP Webcam hoặc DroidCam trên điện thoại.
3. Bật máy chủ video trong ứng dụng.
4. Trong giao diện web, chọn **Điện thoại qua Wi-Fi** và nhập URL ứng dụng cung
   cấp, ví dụ `http://192.168.1.20:8080/video`.

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

Các model được đặt trong thư mục `model/`. `best.pt` phải có class `0` là
`phone`; nếu thay bằng model COCO thì cần cập nhật `PHONE_CLASS_ID` trong
`src/config.py`.
