import cv2

from src.capture import open_webcam, read_frames, release
from src.detector import BiteDetector


def main() -> None:
    cap = open_webcam(device=0)
    print("Press 'q' or ESC to quit.")

    with BiteDetector() as detector:
        try:
            for frame in read_frames(cap):
                result = detector.process(frame)
                cv2.imshow("Bite Detector", result.debug_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 27 = ESC
                    break
        finally:
            release(cap)


if __name__ == "__main__":
    main()
