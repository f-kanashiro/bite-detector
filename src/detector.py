import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from src.utils import euclidean_distance, landmark_to_pixel, draw_alert

# MediaPipe mouth landmark indices (inner lip center approximation)
UPPER_LIP = 13
LOWER_LIP = 14
LIP_LEFT = 78
LIP_RIGHT = 308

# Fingertip landmark indices (MediaPipe Hands)
FINGERTIPS = [4, 8, 12, 16, 20]

# Detection threshold: fraction of face width
NAIL_BITE_THRESHOLD_RATIO = 0.15

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
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
            )
        )
        self._hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(MODELS_DIR / "hand_landmarker.task")
                ),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2,
            )
        )

    def process(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        debug = frame.copy()

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        face_result = self._face_landmarker.detect(mp_image)
        hand_result = self._hand_landmarker.detect(mp_image)

        lip_center: tuple[float, float] | None = None
        face_width: float | None = None

        if face_result.face_landmarks:
            lms = face_result.face_landmarks[0]

            # Draw all face mesh connections (subtle grey)
            for connection in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                p1 = landmark_to_pixel(lms[connection.start], w, h)
                p2 = landmark_to_pixel(lms[connection.end], w, h)
                cv2.line(debug, p1, p2, (80, 80, 80), 1)

            # Highlight key lip landmarks
            for idx in [UPPER_LIP, LOWER_LIP, LIP_LEFT, LIP_RIGHT]:
                px = landmark_to_pixel(lms[idx], w, h)
                cv2.circle(debug, px, 4, (0, 255, 0), -1)

            upper_px = landmark_to_pixel(lms[UPPER_LIP], w, h)
            lower_px = landmark_to_pixel(lms[LOWER_LIP], w, h)
            lip_center = (
                (upper_px[0] + lower_px[0]) / 2,
                (upper_px[1] + lower_px[1]) / 2,
            )

            left_px = landmark_to_pixel(lms[LIP_LEFT], w, h)
            right_px = landmark_to_pixel(lms[LIP_RIGHT], w, h)
            face_width = euclidean_distance(left_px, right_px)

        fingertip_pixels: list[tuple[int, int]] = []
        if hand_result.hand_landmarks:
            for hand_lms in hand_result.hand_landmarks:
                # Draw hand connections
                for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    p1 = landmark_to_pixel(hand_lms[connection.start], w, h)
                    p2 = landmark_to_pixel(hand_lms[connection.end], w, h)
                    cv2.line(debug, p1, p2, (0, 180, 255), 2)

                for tip_idx in FINGERTIPS:
                    px = landmark_to_pixel(hand_lms[tip_idx], w, h)
                    fingertip_pixels.append(px)
                    cv2.circle(debug, px, 6, (255, 0, 0), -1)

        nail_biting = False
        if lip_center and face_width and fingertip_pixels:
            threshold_px = face_width * NAIL_BITE_THRESHOLD_RATIO
            for tip in fingertip_pixels:
                dist = euclidean_distance(tip, lip_center)
                if dist < threshold_px:
                    nail_biting = True
                    break

        if nail_biting:
            draw_alert(debug, "NAIL BITING DETECTED")

        return DetectionResult(nail_biting=nail_biting, debug_frame=debug)

    def close(self) -> None:
        self._face_landmarker.close()
        self._hand_landmarker.close()
