from mddad.realtime_event_manager import DriverSafetyEventManager


def manager():
    return DriverSafetyEventManager(
        session_id=7,
        drowsiness_confirm_seconds=3.0,
        action_confirm_seconds=2.0,
    )


def test_short_signal_never_becomes_event():
    events = manager()
    events.update(
        timestamp=0.0,
        drowsiness_active=True,
        drowsiness_confidence=0.9,
        action_event_type=None,
        action_confidence=None,
        action_label=None,
    )
    update = events.update(
        timestamp=1.0,
        drowsiness_active=False,
        drowsiness_confidence=0.2,
        action_event_type=None,
        action_confidence=None,
        action_label=None,
    )
    assert update.notifications == ()
    assert update.finalized_events == ()
    assert events.confirmed_event_count == 0


def test_drowsiness_is_emitted_once_and_finalized_on_clear():
    events = manager()
    for timestamp in (0.0, 1.0, 2.0):
        update = events.update(
            timestamp=timestamp,
            drowsiness_active=True,
            drowsiness_confidence=0.8 + timestamp * 0.01,
            action_event_type=None,
            action_confidence=None,
            action_label=None,
        )
        assert update.notifications == ()

    update = events.update(
        timestamp=3.0,
        drowsiness_active=True,
        drowsiness_confidence=0.91,
        action_event_type=None,
        action_confidence=None,
        action_label=None,
    )
    assert len(update.notifications) == 1
    assert update.notifications[0].event_type == "drowsiness"
    assert update.drowsiness_confirmed

    repeated = events.update(
        timestamp=4.0,
        drowsiness_active=True,
        drowsiness_confidence=0.89,
        action_event_type=None,
        action_confidence=None,
        action_label=None,
    )
    assert repeated.notifications == ()

    cleared = events.update(
        timestamp=5.0,
        drowsiness_active=False,
        drowsiness_confidence=0.1,
        action_event_type=None,
        action_confidence=None,
        action_label=None,
    )
    assert len(cleared.finalized_events) == 1
    assert cleared.finalized_events[0].duration_seconds == 5.0


def test_action_switch_closes_confirmed_event_and_starts_new_signal():
    events = manager()
    events.update(
        timestamp=0.0,
        drowsiness_active=False,
        drowsiness_confidence=None,
        action_event_type="phone_usage",
        action_confidence=0.7,
        action_label="phone_texting",
    )
    confirmed = events.update(
        timestamp=2.0,
        drowsiness_active=False,
        drowsiness_confidence=None,
        action_event_type="phone_usage",
        action_confidence=0.8,
        action_label="phone_texting",
    )
    assert confirmed.action_confirmed
    assert confirmed.notifications[0].severity == "high"

    switched = events.update(
        timestamp=3.0,
        drowsiness_active=False,
        drowsiness_confidence=None,
        action_event_type="eating_distraction",
        action_confidence=0.75,
        action_label="drinking_eating",
    )
    assert len(switched.finalized_events) == 1
    assert switched.finalized_events[0].event_type == "phone_usage"
    assert switched.finalized_events[0].severity == "high"
    assert not switched.action_confirmed


def test_phone_presence_is_not_an_event_manager_input():
    events = manager()
    update = events.update(
        timestamp=10.0,
        drowsiness_active=False,
        drowsiness_confidence=None,
        action_event_type=None,
        action_confidence=None,
        action_label=None,
    )
    assert update.active_event_types == ()
    assert events.confirmed_event_count == 0
