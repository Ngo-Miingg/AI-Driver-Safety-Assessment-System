from mddad.face_behavior_signals import FaceBehaviorSignalExtractor
from mddad.realtime_event_manager import DriverSafetyEventManager


class SignalConfig:
    config = {
        "MAR_THRESHOLD": 0.60,
        "FACE_LOOKING_AWAY_OFFSET_THRESHOLD": 0.07,
        "FACE_HEAD_DOWN_OFFSET_THRESHOLD": 0.08,
        "FACE_BEHAVIOR_CALIBRATION_SAMPLES": 5,
    }


def _geometry(mar=0.2, horizontal=0.0, vertical=0.0, yaw=None):
    result = {
        "mar": mar,
        "horizontal_offset": horizontal,
        "vertical_offset": vertical,
    }
    if yaw is not None:
        result["yaw_degrees"] = yaw
    return result


def test_held_head_turn_is_looking_away_even_without_motion():
    config = SignalConfig()
    config.config.update({
        "FACE_LOOKING_AWAY_YAW_THRESHOLD_DEGREES": 20.0,
        "FACE_LOOKING_AWAY_YAW_RELEASE_DEGREES": 15.0,
    })
    extractor = FaceBehaviorSignalExtractor(config)
    for _ in range(5):
        extractor.observe(_geometry(yaw=0.0))

    state = extractor.observe(_geometry(yaw=24.0))
    held = extractor.observe(_geometry(yaw=23.0))
    returned = extractor.observe(_geometry(yaw=10.0))

    assert state["looking_away"].active
    assert held["looking_away"].active
    assert not returned["looking_away"].active


def test_neutral_calibration_prevents_false_head_events():
    extractor = FaceBehaviorSignalExtractor(SignalConfig())
    for _ in range(5):
        state = extractor.observe(_geometry(horizontal=0.02, vertical=0.03))

    assert state["calibrated"]
    assert not state["looking_away"].active
    assert not state["head_down"].active


def test_three_signals_are_independent_after_calibration():
    extractor = FaceBehaviorSignalExtractor(SignalConfig())
    for _ in range(5):
        extractor.observe(_geometry())

    yawning = extractor.observe(_geometry(mar=0.75))
    looking = extractor.observe(_geometry(horizontal=-0.10))
    down = extractor.observe(_geometry(vertical=0.11))

    assert yawning["yawning"].active
    assert not yawning["looking_away"].active
    assert looking["looking_away"].active
    assert not looking["head_down"].active
    assert down["head_down"].active


def test_face_signal_requires_continuous_temporal_confirmation():
    manager = DriverSafetyEventManager(
        session_id=11,
        drowsiness_confirm_seconds=3.0,
        action_confirm_seconds=2.0,
        yawn_confirm_seconds=1.2,
    )
    base = {
        "drowsiness_active": False,
        "drowsiness_confidence": None,
        "action_event_type": None,
        "action_confidence": None,
        "action_label": None,
    }
    manager.update(
        timestamp=0.0,
        face_signals={"yawn": {"active": True, "confidence": 0.7}},
        **base,
    )
    short = manager.update(
        timestamp=1.0,
        face_signals={"yawn": {"active": True, "confidence": 0.8}},
        **base,
    )
    confirmed = manager.update(
        timestamp=1.2,
        face_signals={"yawn": {"active": True, "confidence": 0.9}},
        **base,
    )

    assert short.notifications == ()
    assert confirmed.face_confirmed == ("yawn",)
    assert confirmed.notifications[0].event_type == "yawn"
