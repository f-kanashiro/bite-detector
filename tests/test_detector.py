from src.utils import euclidean_distance, landmark_to_pixel


class _FakeLandmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def test_euclidean_distance_3_4_5() -> None:
    assert euclidean_distance((0, 0), (3, 4)) == 5.0


def test_euclidean_distance_same_point() -> None:
    assert euclidean_distance((7, 3), (7, 3)) == 0.0


def test_landmark_to_pixel_center() -> None:
    lm = _FakeLandmark(0.5, 0.5)
    assert landmark_to_pixel(lm, 640, 480) == (320, 240)


def test_landmark_to_pixel_origin() -> None:
    lm = _FakeLandmark(0.0, 0.0)
    assert landmark_to_pixel(lm, 640, 480) == (0, 0)


def test_landmark_to_pixel_full() -> None:
    lm = _FakeLandmark(1.0, 1.0)
    assert landmark_to_pixel(lm, 640, 480) == (640, 480)
