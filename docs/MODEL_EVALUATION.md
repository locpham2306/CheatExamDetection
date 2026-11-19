# Đánh giá model cho demo chống gian lận

## Kết luận triển khai

Không dùng một model duy nhất để kết luận gian lận. Pipeline hiện tại tách
phát hiện người, pose/hướng đầu và điện thoại; sau đó liên kết theo ID và kiểm
tra hành vi theo thời gian.

- Quay đầu: video lớp học được xử lý theo hai tile chồng lấn; cùng một người
  phải có ít nhất ba mẫu liên tiếp cùng hướng và giữ trái/phải ít nhất 2 giây.
- Điện thoại: model chuyên dụng phải thấy điện thoại qua nhiều lần suy luận và
  điện thoại phải nằm gần một người đang được theo dõi.
- Tài liệu: COCO `book` hiện chỉ là candidate. Cảnh báo trao đổi chỉ xuất hiện
  khi candidate được theo dõi và đổi chủ ổn định.

Tối ưu runtime hiện tại: ONNX Runtime Web được yêu cầu dùng GPU hiệu năng cao
và warm-up model điện thoại trước khi cho bắt đầu. Không xếp hàng frame: mỗi
lần chỉ xử lý một frame điện thoại, khi xong sẽ lấy frame mới nhất theo nhịp
250 ms (tối đa 4 lần/giây). Lượt phone có khe GPU riêng, xen kẽ với model
người/pose để tránh ba session tranh hàng. Cảnh từ 8 người được chia thành bốn
tile chồng lấn 56% × 58% và bật thêm model COCO; cảnh gần dùng model custom
toàn khung để phản hồi nhanh. Khung điện thoại hiện ngay, còn nhật ký cần tồn
tại 0,5 giây ở cảnh gần hoặc 1 giây ở lớp học và phải gắn được với ID người.

## Kiểm tra trên dữ liệu của dự án

Model `best_phone_detection.pt` của dự án được xuất thành ONNX và kiểm tra với
cùng phép resize 640×640 mà browser sử dụng:

- Ảnh iPhone cầm trên tay trong `Smart-Proctor-System`: phát hiện `phone` 79%.
- 12 frame rải đều trong video lớp học Altima: 11 frame không báo điện thoại;
  một candidate 32%. Đây là kiểm tra smoke, chưa phải tập test gán nhãn.
- Browser end-to-end: điện thoại được gắn với đúng ID người và chỉ cảnh báo sau
  khi tồn tại qua nhiều lần inference.
- Đo lại đoạn 20–28 giây của Altima sau khi tile: số pose có đủ keypoint mặt
  tăng từ khoảng 10 lên 23/24; cảnh báo quay đầu giả giảm từ 6 xuống 0.

Pipeline browser hiện ghép thêm `yolov8n-coco-phone.onnx`, chỉ lấy class COCO
`cell phone` (67), sau đó NMS chung với model chuyên dụng. Kiểm tra trên ba ảnh
COCO có điện thoại cho thấy hai model bù nhau: ở một ảnh model dự án chỉ đạt
1,3% trong khi model COCO đạt 25,6%; ở ảnh khác model dự án đạt 76,8% còn model
COCO đạt 34,5%. Smoke test end-to-end bằng video WebM xác nhận model COCO bắt
được điện thoại 55,9%, kết quả được đổi về tọa độ khung gốc và gắn đúng ID
người. Đây là kiểm tra tích hợp, chưa thay thế tập test có gán nhãn của dự án.

Model COCO được lưu ở dạng ONNX (không nạp pickle), SHA-256
`65158dad735be799c2466fa15e260c09558080bd530b42a8d0c3d1b419afd8b5`.
Nguồn và lưu ý license nằm trong `web_proctor/models/README.md`.

Ngày 2026-11-19, model custom chính được thay bằng checkpoint do chủ dự án tự
train (`best_phone_predict.pt`), gồm `cell_phone` và `calculator`. Dashboard
chỉ giải mã class `cell_phone` rồi ghép với model COCO. Bản ONNX dùng input
640×640, opset 17; checksum và cấu trúc output được ghi trong
`web_proctor/models/README.md`. Đồng thời bỏ giới hạn hậu xử lý cũ chỉ cho phép
điện thoại chiếm tối đa 1,5% khung hình vì giới hạn này làm điện thoại cầm sát
webcam bị loại dù detector đã nhận ra.

Smoke test cho thấy model custom mới vẫn có thể nhầm thẻ tên, tay và mảng chữ
trên áo thành điện thoại. Vì vậy kết quả custom-only ở cảnh đông dùng ngưỡng
cao hơn và kiểm tra tỷ lệ cạnh; nếu custom và COCO chồng bbox thì được đánh dấu
`phone-consensus` và dùng ngưỡng thấp hơn. Cảnh báo cuối vẫn cần gắn bbox với
một người và tồn tại liên tục theo thời gian.

Model `srijannn/aiproctor-yolo-detector` không được tích hợp. Model card công
bố precision 13,9%, mAP50 25,9%; benchmark trên 12 frame Altima ở ngưỡng 0,15
cho 76 candidate điện thoại. File thử nghiệm đã được xóa.

## Nguồn đã rà

- [IndUSV/yolov8n-mobile-phone](https://huggingface.co/IndUSV/yolov8n-mobile-phone):
  MIT, nhỏ khoảng 6,2 MB và nhanh, nhưng validation thể hiện số mẫu rất ít
  (recall thay đổi theo bước khoảng 1/11). Kho weights là PyTorch pickle và
  Hugging Face liệt kê 24 pickle imports; chưa nạp vào máy, chưa đủ cơ sở thay
  model nội bộ 44,7 MB.
- [Classroom Cell Phone Detection](https://universe.roboflow.com/hello-tlxm1/classroom-cell-phone-detection-fwf4b-bzuea):
  đúng bối cảnh lớp học nhưng chỉ 253 ảnh; model công bố mAP50 42,6%, precision
  55,7% và recall 51,5%, chưa tốt hơn mức cần cho demo.
- [Student Classroom Activity v2](https://universe.roboflow.com/studentactivity/student-classroom-activity-v-2):
  CC BY 4.0, 1.094 ảnh với ba lớp phone/sleep/study. Phù hợp làm nguồn bổ sung
  khi fine-tune, nhưng vẫn phải ghép dữ liệu quay từ đúng iPhone/phòng demo.

- [Mobile Usage Detection (Roboflow)](https://universe.roboflow.com/rose-epnzt/mobile-usage-detection-so4ig):
  chỉ một class phone và chỉ số validation cao, nhưng tập khá nhỏ; cần tải
  dataset và đánh giá lại trên góc camera của dự án.
- [ExamGuardian AI (Roboflow)](https://universe.roboflow.com/tri-le-ky44w/examguardian-ai-vnmur):
  có `phone` và `suspicious_paper`, phù hợp nhất để bổ sung dữ liệu giấy.
- [SentinelVision](https://huggingface.co/Kawal511/sentinelvision-proctoring-yolov8n):
  có phone/book/headphone/laptop/person nhưng chính tác giả báo 56,4% false
  positive trên webcam chưa thấy; không dùng thẳng vào demo.
- [Cheatomaly](https://pmc.ncbi.nlm.nih.gov/articles/PMC13201167/): 94 video,
  có phone, paper sheet, look around và passing paper. Dataset xử lý/annotation
  chỉ cấp khi liên hệ tác giả.
- [Cheating Scenario Dataset](https://data.mendeley.com/datasets/mjrfmvsh7d/1):
  CC BY 4.0, 15.051 ảnh về head pose/eye gaze; hữu ích nếu cần thay heuristic
  hướng đầu, nhưng không giải quyết điện thoại/trao giấy.
- [Invigilo](https://github.com/chandan-srinivas29031998/Invigilo): kiến trúc
  ByteTrack + pose + cửa sổ thời gian đáng tham khảo; dữ liệu video tự quay
  không được phát hành đầy đủ và báo cáo vẫn có nhiều false flag hard-negative.

Không nạp các file PyTorch `.pt` tải tùy ý từ Internet. `.pt` dựa trên pickle
có thể chạy mã khi load. Chỉ dùng model nội bộ đã biết nguồn, hoặc ưu tiên ONNX
và kiểm tra trên tập test riêng trước khi tích hợp.

## Muốn tốt hơn mức demo

1. Quay tối thiểu 30–50 clip cho mỗi hành vi ở đúng camera/phòng sẽ demo:
   bình thường, cầm điện thoại, điện thoại nằm bàn, quay trái/phải 1 giây và 3
   giây, lật trang, đưa giấy, nhận giấy. Thêm hard-negative như cầm bút, máy
   tính, gãi đầu và cúi viết.
2. Gán nhãn `phone`, `paper`, `hand` và ID người; tách train/validation/test
   theo buổi quay hoặc theo người, không chia ngẫu nhiên các frame cùng clip.
3. Fine-tune detector trên dữ liệu Roboflow cộng dữ liệu thật của dự án. Chọn
   threshold từ precision/recall trên tập test, không chọn theo video demo.
4. Đo event-level precision/recall: một hành vi kéo dài là một event, không
   tính từng frame như một dự đoán độc lập.
5. Luôn giữ ảnh/video bằng chứng để người giám thị xác nhận; hệ thống chỉ gắn
   cờ nghi vấn.

## Video benchmark cục bộ

Hai video MIT dung lượng tổng khoảng 12,5 MB đã được tải vào `test_videos/`.
Nguồn, kịch bản, kết quả smoke test và lệnh tải lại nằm trong
`test_videos/README.md`. Các file MP4 được ignore khỏi Git.
