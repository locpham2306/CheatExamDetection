# Exam Cheat Detection

Demo phát hiện hành vi đáng ngờ trong phòng thi bằng camera. Hệ thống giữ model
YOLO cũ để nhận diện/tracking người, dùng hai model ensemble để phát hiện điện
thoại và dùng pose keypoint để ước lượng hướng đầu.

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
File**. Launcher này mở dashboard WebGPU mới trong `web_proctor/`. Hoặc chạy lệnh:

```powershell
python src/app.py
```

Lệnh trên tự mở giao diện giám sát camera điện thoại, webcam máy tính hoặc video
tải lên. Giao diện Streamlit cũ chỉ còn để tham khảo và có thể chạy trực tiếp:

```powershell
streamlit run src/web.py
```

Giao diện WebRTC cũ vẫn có thể chạy riêng bằng:

```powershell
streamlit run src/webrtc_app.py
```

### Web proctor chạy trực tiếp trên trình duyệt

Source `ai-exam-proctor` được đặt trong `web_proctor/`, kèm sẵn model YOLO,
MediaPipe Face Landmarker và ONNX Runtime. Chạy trên Windows bằng:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_web_proctor.ps1
```

Trình duyệt sẽ tự mở tại `http://localhost:8000`. Giao diện có hai màn riêng:
**Camera trực tiếp** và **Phân tích video**. Trong màn camera, chỉ cần chọn
**Camera điện thoại (DroidCam)** hoặc **Camera máy tính**, rồi nhấn
**Bắt đầu giám sát**. Chế độ này khoanh người, hiển thị hướng quay đầu
và cảnh báo điện thoại/tài liệu trực tiếp trên hình ảnh. Nhật ký bên phải lưu
thời gian, nội dung và ảnh của từng cảnh báo; có thể tải từng ảnh hoặc nhấn
**Xuất minh chứng** để lấy toàn bộ dưới dạng ZIP. Lần mở đầu tiên cần chờ bốn
model ONNX được nạp từ ổ đĩa. Không cần cài thêm thư viện Python.

Khi DroidCam Client đã kết nối iPhone và hiện hình, chọn **Camera điện thoại
(DroidCam)** rồi nhấn **Làm mới**. Ứng dụng tự tìm driver có tên `DroidCam` ở
phía sau; không cần chọn từ danh sách driver. Có thể chọn 1080p, 720p hoặc 480p,
nhưng độ phân giải thực tế vẫn phụ thuộc thiết lập của DroidCam. Tên camera, độ
phân giải và FPS thực tế sẽ hiện ngay dưới nút bắt đầu.

Với video lớp học, giao diện dùng thêm model pose của dự án
`yolov8n-pose.onnx`: YOLO khoanh người, còn các keypoint mũi–mắt–tai được dùng
để xác định trái/phải cho từng khung. MediaPipe chỉ bổ trợ khi dùng webcam cận
mặt. Video góc rộng được chia thành hai tile chồng lấn và xử lý luân phiên để
đầu người ở xa có nhiều pixel hơn; kết quả được ghép lại vào khung gốc.

Pipeline browser hiện tách thành ba phần:

- `yolo26s.onnx`: người, laptop và tài liệu dạng sách.
- `yolov8n-pose.onnx`: keypoint cơ thể/mặt và hướng đầu.
- Hai model điện thoại chạy dạng ensemble: `best_phone_predict.onnx` do dự án
  tự train (chỉ lấy class `cell_phone`, bỏ class `calculator`) và
  `yolov8n-coco-phone.onnx` lọc class COCO `cell phone` (67). Kết quả được
  ghép bằng NMS để model này bù ca bỏ sót của model kia, rồi mới gắn điện thoại
  với ID người gần nhất.

Các worker ONNX ưu tiên WebGPU ở chế độ `high-performance`; trên máy có GPU rời
không cần cài thêm CUDA/PyTorch cho bản chạy trình duyệt. Với cảnh lớp học,
detector điện thoại quét luân phiên bốn vùng chồng lấn thay vì chỉ thu nhỏ cả
lớp về 640×640. Cách này làm điện thoại nhỏ lớn hơn trong đầu vào model, đồng
thời giữ kết quả các vùng trong một cửa sổ ngắn để tracking không bị chớp.
Thẻ **Điện thoại / GPU** trên giao diện hiển thị `2 MODEL`, backend, thời gian
inference thực tế và số candidate của từng model.

Mỗi người có ID ổn định giữa các frame. Quay trái/phải chỉ chuyển thành cảnh
báo đỏ khi cùng một ID có ít nhất ba kết quả liên tiếp cùng hướng và giữ hướng
ít nhất 2 giây; một frame lệch không còn tạo cảnh báo. Tracker giữ ID tối đa 7
giây khi detector chớp mất. Điện thoại phải được gắn với người gần nhất và
xuất hiện qua nhiều lần suy luận. Class COCO `book` chỉ được coi là **tài liệu
nghi vấn**; sự kiện trao đổi chỉ được tạo khi cùng một tài liệu vừa đổi người
sở hữu vừa di chuyển đủ xa. Đây là tín hiệu để giám thị xem lại, không phải
bằng chứng tự động kết luận gian lận.

Phần lọc điện thoại kết hợp quy tắc từ Smart-Proctor-System: kiểm tra độ tin
cậy, tỷ lệ khung, kích thước và độ chồng lấn với remote/book/laptop để giảm
nhận nhầm vật thể trong lớp học.

Để phân tích video có sẵn, nhấn **Chọn video**, chọn file MP4/WebM/MOV rồi
nhấn **Phân tích video**. File được xử lý cục bộ trong trình duyệt, không tải
lên máy chủ.

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

Các model ONNX tải từ mạng có thể benchmark an toàn trên video trước khi đưa
vào app:

```powershell
python tools/benchmark_onnx.py path\to\model.onnx path\to\video.mp4 --samples 12
```

Kết quả rà model/dataset và kế hoạch cải thiện tiếp theo nằm trong
`docs/MODEL_EVALUATION.md`.
