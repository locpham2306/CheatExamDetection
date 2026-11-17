# Model files

## `yolov8n-coco-phone.onnx`

- Source: <https://huggingface.co/Kalray/yolov8n>
- Downloaded file: `yolov8n.onnx`
- Task: YOLOv8n object detection trained on COCO; this project decodes only
  class `67` (`cell phone`).
- SHA-256: `65158dad735be799c2466fa15e260c09558080bd530b42a8d0c3d1b419afd8b5`
- Model card license: GPL-3.0. The model card references the Ultralytics YOLOv8
  implementation; review upstream terms before redistributing commercially.

The browser loads this ONNX file together with `best_phone_detection.onnx`.
It does not load a remote PyTorch/pickle checkpoint at runtime.
