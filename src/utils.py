import numpy as np
import cv2
from typing import Protocol


class _Landmark(Protocol):
    x: float
    y: float


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def landmark_to_pixel(
    landmark: _Landmark, frame_width: int, frame_height: int
) -> tuple[int, int]:
    """Convert a normalized MediaPipe landmark to pixel coordinates."""
    return int(landmark.x * frame_width), int(landmark.y * frame_height)


def draw_alert(frame: np.ndarray, message: str) -> None:
    """Overlay a red alert message at the top of the frame (modifies in-place)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame,
        message,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
