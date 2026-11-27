from __future__ import annotations


def filter_phone_detections(detections, frame_shape, face_box, enabled=True):
    """Keep phone boxes that are geometrically plausible for the visible driver.

    The detector confidence is intentionally left untouched. This filter only
    rejects boxes that are outside the driver's interaction area or implausible
    in size relative to the detected face.
    """
    phone_detections = [item for item in detections if item.get("is_phone")]
    if not enabled or not face_box:
        return phone_detections

    frame_height, frame_width = frame_shape[:2]
    fx1, fy1, fx2, fy2 = [float(value) for value in face_box]
    face_width = max(fx2 - fx1, 1.0)
    face_height = max(fy2 - fy1, 1.0)
    face_center_x = (fx1 + fx2) / 2.0
    face_center_y = (fy1 + fy2) / 2.0
    face_area = face_width * face_height

    accepted = []
    for detection in phone_detections:
        bbox = detection.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        box_width = max(x2 - x1, 0.0)
        box_height = max(y2 - y1, 0.0)
        if box_width <= 0.0 or box_height <= 0.0:
            continue

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        relative_width = box_width / face_width
        relative_height = box_height / face_height
        relative_area = (box_width * box_height) / face_area

        # A handheld phone should be close to the driver's face/upper body.
        normalized_x = abs(center_x - face_center_x) / face_width
        normalized_y = (center_y - face_center_y) / face_height
        in_driver_interaction_zone = normalized_x <= 1.9 and -0.85 <= normalized_y <= 2.8

        # Reject scene rectangles (windows, monitors, wall panels) while keeping
        # portrait/landscape phones at normal cabin-camera distances.
        plausible_size = (
            0.06 <= relative_width <= 1.35
            and 0.10 <= relative_height <= 1.90
            and relative_area <= 1.55
        )

        edge_count = sum(
            (
                x1 <= 1.0,
                y1 <= 1.0,
                x2 >= frame_width - 1.0,
                y2 >= frame_height - 1.0,
            )
        )
        frame_area_ratio = (box_width * box_height) / max(float(frame_width * frame_height), 1.0)
        looks_like_scene_region = edge_count >= 2 or (edge_count >= 1 and frame_area_ratio >= 0.08)

        if in_driver_interaction_zone and plausible_size and not looks_like_scene_region:
            accepted.append(detection)

    return accepted
