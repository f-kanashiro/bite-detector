import cv2


def main() -> None:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    print("Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to read frame.")
            break

        cv2.imshow("Bite Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27 = ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
