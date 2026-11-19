# Model files

## `best_phone_predict.onnx`

- Source: checkpoint `best_phone_predict.pt` trained and supplied by the
  project owner on 2026-11-19.
- Task: YOLO detection with classes `0: cell_phone`, `1: calculator`; the
  browser decoder uses only class `0`.
- Export: ONNX opset 17, static input `1×3×640×640`, output `1×6×8400`.
- Checkpoint SHA-256:
  `4a6410d46687589cd4628b04141d5d4879e3b695b07f2316988d12cb6ca84ed3`.
- ONNX SHA-256:
  `037fd1a04372fb618fac55b02b142ca0640289281f9f4c49518a016d334ccd02`.

This is the primary custom phone detector. The browser combines it with the
COCO detector below and merges overlapping results with NMS.

## `yolov8n-coco-phone.onnx`

- Source: <https://huggingface.co/Kalray/yolov8n>
- Downloaded file: `yolov8n.onnx`
- Task: YOLOv8n object detection trained on COCO; this project decodes only
  class `67` (`cell phone`).
- SHA-256: `65158dad735be799c2466fa15e260c09558080bd530b42a8d0c3d1b419afd8b5`
- Model card license: GPL-3.0. The model card references the Ultralytics YOLOv8
  implementation; review upstream terms before redistributing commercially.

The browser loads this ONNX file together with `best_phone_predict.onnx`.
It does not load a remote PyTorch/pickle checkpoint at runtime.
