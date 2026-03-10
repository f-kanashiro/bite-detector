import cv2
from typing import Generator
import numpy as np


def open_webcam(device: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam device {device}.")
    return cap


def read_frames(cap: cv2.VideoCapture) -> Generator[np.ndarray, None, None]:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def release(cap: cv2.VideoCapture) -> None:
    cap.release()
    cv2.destroyAllWindows()
