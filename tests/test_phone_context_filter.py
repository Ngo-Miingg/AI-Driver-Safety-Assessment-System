from mddad.phone_context_filter import filter_phone_detections


def _phone(bbox, confidence=0.35):
    return {
        "class_name": "Mobile-phone",
        "confidence": confidence,
        "bbox": bbox,
        "is_phone": True,
    }


def test_keeps_phone_near_driver_without_changing_confidence():
    detection = _phone([390, 250, 455, 380], confidence=0.36)

    result = filter_phone_detections(
        [detection],
        (480, 640, 3),
        (250, 90, 390, 250),
    )

    assert result == [detection]
    assert result[0]["confidence"] == 0.36


def test_rejects_large_background_rectangle_even_at_high_confidence():
    result = filter_phone_detections(
        [_phone([5, 20, 620, 410], confidence=0.97)],
        (480, 640, 3),
        (250, 90, 390, 250),
    )

    assert result == []


def test_rejects_phone_like_background_object_far_from_driver():
    result = filter_phone_detections(
        [_phone([15, 30, 80, 150], confidence=0.82)],
        (480, 640, 3),
        (280, 110, 380, 230),
    )

    assert result == []


def test_filter_can_be_disabled_for_detector_diagnostics():
    detection = _phone([15, 30, 80, 150], confidence=0.82)

    result = filter_phone_detections(
        [detection],
        (480, 640, 3),
        (280, 110, 380, 230),
        enabled=False,
    )

    assert result == [detection]
