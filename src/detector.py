import cv2
import mediapipe as mp
import numpy as np
import time
from dataclasses import dataclass
from pathlib import Path

from src.utils import euclidean_distance, landmark_to_pixel, draw_alert

# MediaPipe mouth landmark indices
UPPER_LIP = 13
LOWER_LIP = 14
LIP_LEFT = 78
LIP_RIGHT = 308

# Outer lip corners and top/bottom for a broader mouth region
MOUTH_LANDMARKS = [13, 14, 78, 308, 82, 312, 87, 317, 95, 324, 185, 409]

# Hand landmark indices: fingertips + DIP joints (one segment below each tip)
# Thumb: 3=IP, 4=TIP | Index: 7=DIP, 8=TIP | Middle: 11=DIP, 12=TIP
# Ring: 15=DIP, 16=TIP | Pinky: 19=DIP, 20=TIP
FINGER_LANDMARKS = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]

# Detection threshold: fraction of face width
NAIL_BITE_THRESHOLD_RATIO = 0.35

MODELS_DIR = Path(__file__).parent.parent / "models"


@dataclass
class DetectionResult:
    nail_biting: bool
    debug_frame: np.ndarray


class BiteDetector:
    def __init__(self) -> None:
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(MODELS_DIR / "face_landmarker.task")
                ),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1,
            )
        )
        self._hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(MODELS_DIR / "hand_landmarker.task")
                ),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
            )
        )
        self._start_ms = time.monotonic_ns() // 1_000_000

    def process(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        debug = frame.copy()

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        timestamp_ms = time.monotonic_ns() // 1_000_000 - self._start_ms
        face_result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        mouth_pixels: list[tuple[int, int]] = []
        face_width: float | None = None

        if face_result.face_landmarks:
            lms = face_result.face_landmarks[0]

            # Draw all face mesh connections (subtle grey)
            for connection in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                p1 = landmark_to_pixel(lms[connection.start], w, h)
                p2 = landmark_to_pixel(lms[connection.end], w, h)
                cv2.line(debug, p1, p2, (80, 80, 80), 1)

            # Collect and highlight mouth region landmarks
            for idx in MOUTH_LANDMARKS:
                px = landmark_to_pixel(lms[idx], w, h)
                mouth_pixels.append(px)
                cv2.circle(debug, px, 4, (0, 255, 0), -1)

            left_px = landmark_to_pixel(lms[LIP_LEFT], w, h)
            right_px = landmark_to_pixel(lms[LIP_RIGHT], w, h)
            face_width = euclidean_distance(left_px, right_px)

        finger_pixels: list[tuple[int, int]] = []
        if hand_result.hand_landmarks:
            for hand_lms in hand_result.hand_landmarks:
                # Draw hand connections
                for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    p1 = landmark_to_pixel(hand_lms[connection.start], w, h)
                    p2 = landmark_to_pixel(hand_lms[connection.end], w, h)
                    cv2.line(debug, p1, p2, (0, 180, 255), 2)

                for lm_idx in FINGER_LANDMARKS:
                    px = landmark_to_pixel(hand_lms[lm_idx], w, h)
                    finger_pixels.append(px)
                    cv2.circle(debug, px, 6, (255, 0, 0), -1)

        nail_biting = False
        if mouth_pixels and face_width is not None and finger_pixels:
            threshold_px = face_width * NAIL_BITE_THRESHOLD_RATIO
            nail_biting = any(
                euclidean_distance(fp, mp_) < threshold_px
                for fp in finger_pixels
                for mp_ in mouth_pixels
            )

        if nail_biting:
            draw_alert(debug, "NAIL BITING DETECTED")

        return DetectionResult(nail_biting=nail_biting, debug_frame=debug)

    def __enter__(self) -> "BiteDetector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._face_landmarker.close()
        self._hand_landmarker.close()
