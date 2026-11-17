"""Benchmark the two browser phone detectors on local images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def output_rows(raw: np.ndarray) -> np.ndarray:
    rows = np.asarray(raw).squeeze(0)
    if rows.ndim != 2:
        raise ValueError(f"Unsupported output: {raw.shape}")
    return rows.T if rows.shape[0] <= 256 and rows.shape[0] < rows.shape[1] else rows


def scores(raw: np.ndarray, class_id: int) -> list[float]:
    rows = output_rows(raw)
    score_index = 4 + class_id
    if score_index >= rows.shape[1]:
        return []
    return sorted((float(value) for value in rows[:, score_index]), reverse=True)


def infer(net: cv2.dnn.Net, image: np.ndarray) -> np.ndarray:
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (640, 640), swapRB=True, crop=False
    )
    net.setInput(blob)
    return net.forward()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--custom", type=Path,
        default=Path("web_proctor/models/best_phone_detection.onnx"),
    )
    parser.add_argument(
        "--coco", type=Path,
        default=Path("web_proctor/models/yolov8n-coco-phone.onnx"),
    )
    args = parser.parse_args()

    custom_net = cv2.dnn.readNetFromONNX(str(args.custom))
    coco_net = cv2.dnn.readNetFromONNX(str(args.coco))
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path.name}: unreadable")
            continue
        custom = scores(infer(custom_net, image), 0)[:3]
        coco = scores(infer(coco_net, image), 67)[:3]
        print(
            f"{path.name}: custom={','.join(f'{x:.3f}' for x in custom)} "
            f"coco={','.join(f'{x:.3f}' for x in coco)}"
        )


if __name__ == "__main__":
    main()
