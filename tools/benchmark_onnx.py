"""Quick, reproducible ONNX detector check on a local video.

This intentionally uses OpenCV DNN instead of loading external PyTorch pickle
files. It prints detections only and never modifies the source video.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


EXAM_CLASSES = {
    0: "cell_phone",
    1: "book",
    2: "paper_notes",
    3: "calculator",
    4: "secondary_screen",
}

PHONE_ONLY_CLASSES = {0: "phone"}

PHONE_CALCULATOR_CLASSES = {
    0: "cell_phone",
    1: "calculator",
}

COCO_RELEVANT_CLASSES = {
    0: "person",
    63: "laptop",
    67: "cell_phone",
    73: "book",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--samples", type=int, default=8)
    return parser.parse_args()


def normalize_output(raw: np.ndarray) -> np.ndarray:
    output = np.asarray(raw).squeeze(0)
    if output.ndim != 2:
        raise ValueError(f"Unsupported output shape: {raw.shape}")
    if output.shape[0] < output.shape[1] and output.shape[0] <= 256:
        output = output.T
    return output


def decode(raw: np.ndarray, threshold: float) -> list[tuple[int, float, list[int]]]:
    predictions = normalize_output(raw)
    class_count = predictions.shape[1] - 4
    if class_count <= 0:
        raise ValueError(f"Expected YOLO output, got {predictions.shape}")

    boxes: list[list[int]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    for row in predictions:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < threshold:
            continue
        cx, cy, width, height = map(float, row[:4])
        boxes.append([
            int(cx - width / 2),
            int(cy - height / 2),
            int(width),
            int(height),
        ])
        scores.append(confidence)
        class_ids.append(class_id)

    kept: list[tuple[int, float, list[int]]] = []
    for class_id in sorted(set(class_ids)):
        indices = [index for index, value in enumerate(class_ids) if value == class_id]
        class_boxes = [boxes[index] for index in indices]
        class_scores = [scores[index] for index in indices]
        selected = cv2.dnn.NMSBoxes(class_boxes, class_scores, threshold, 0.45)
        for selected_index in np.asarray(selected).reshape(-1):
            original_index = indices[int(selected_index)]
            kept.append((class_id, scores[original_index], boxes[original_index]))
    return kept


def main() -> None:
    args = parse_args()
    net = cv2.dnn.readNetFromONNX(str(args.model))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    duration = frame_count / max(fps, 1.0)
    sample_times = np.linspace(0, max(0.0, duration - 0.05), max(1, args.samples))
    totals: Counter[str] = Counter()

    for second in sample_times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(640, 640),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        raw = net.forward()
        detections = decode(raw, args.confidence)
        class_count = normalize_output(raw).shape[1] - 4
        labels = (
            PHONE_ONLY_CLASSES if class_count == 1
            else PHONE_CALCULATOR_CLASSES if class_count == 2
            else EXAM_CLASSES if class_count == 5
            else COCO_RELEVANT_CLASSES
        )
        readable = []
        for class_id, confidence, _ in detections:
            label = labels.get(class_id, f"class_{class_id}")
            if class_count not in (1, 2, 5) and class_id not in COCO_RELEVANT_CLASSES:
                continue
            totals[label] += 1
            readable.append(f"{label}:{confidence:.2f}")
        print(f"{second:6.2f}s | {', '.join(readable) or '-'}")

    capture.release()
    print("TOTAL | " + ", ".join(f"{key}:{value}" for key, value in totals.items()))


if __name__ == "__main__":
    main()
