from __future__ import annotations

from .analysis_models import Event, RiskState, SessionAnalysisSummary


EVENT_CATEGORY = {
    "normal_driving": "visibility",
    "phone_interaction": "phone_usage",
    "phone_calling": "phone_usage",
    "phone_texting": "phone_usage",
    "phone_use": "phone_usage",
    "operating_radio": "distraction",
    "drinking_eating": "distraction",
    "reaching_behind": "distraction",
    "hair_and_makeup": "distraction",
    "in_vehicle_display_interaction": "distraction",
    "eating_distraction": "distraction",
    "drinking_distraction": "distraction",
    "visual_posture_distraction": "distraction",
    "eye_closure": "drowsiness",
    "drowsiness": "drowsiness",
    "yawning": "drowsiness",
    "yawn": "drowsiness",
    "phone_usage": "phone_usage",
    "phone_call": "phone_usage",
    "talking_to_passenger": "distraction",
    "looking_away": "distraction",
    "head_down": "distraction",
    "mixed_risk": "distraction",
    "face_missing": "visibility",
}

BASE_WEIGHTS = {
    "normal_driving": 0,
    "phone_interaction": 22,
    "phone_calling": 25,
    "phone_texting": 25,
    "phone_use": 25,
    "operating_radio": 14,
    "drinking_eating": 16,
    "reaching_behind": 18,
    "hair_and_makeup": 14,
    "in_vehicle_display_interaction": 18,
    "eating_distraction": 14,
    "drinking_distraction": 12,
    "visual_posture_distraction": 16,
    "eye_closure": 18,
    "drowsiness": 18,
    "yawning": 10,
    "yawn": 10,
    "phone_usage": 25,
    "phone_call": 25,
    "talking_to_passenger": 12,
    "looking_away": 16,
    "head_down": 20,
    "mixed_risk": 28,
    "face_missing": 8,
}

SEVERITY_MULTIPLIERS = {
    "low": 0.6,
    "medium": 1.0,
    "high": 1.5,
    "critical": 2.2,
}

CATEGORIES = ("drowsiness", "distraction", "phone_usage", "visibility")


def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, value))


def alert_level_for_score(score):
    if score >= 75:
        return "Danger"
    if score >= 50:
        return "Warning"
    if score >= 25:
        return "Caution"
    return "Normal"


def contribution_for_event(event):
    base = BASE_WEIGHTS.get(event.event_type, 5)
    severity_multiplier = SEVERITY_MULTIPLIERS.get(event.severity, 1.0)
    duration_multiplier = min(1.0 + event.duration_seconds / 10.0, 2.0)
    confidence_multiplier = max(0.5, min(event.confidence or 1.0, 1.0))
    return round(base * severity_multiplier * duration_multiplier * confidence_multiplier, 2)


def categorize_contributions(events):
    totals = {category: 0.0 for category in CATEGORIES}
    for event in events:
        category = EVENT_CATEGORY.get(event.event_type, "visibility")
        totals[category] += contribution_for_event(event)
    return {category: round(value, 2) for category, value in totals.items()}


def percentage_breakdown(category_totals):
    total = sum(category_totals.values())
    if total <= 0:
        return {category: 0.0 for category in CATEGORIES}
    return {
        category: round((value / total) * 100, 1)
        for category, value in category_totals.items()
    }


def build_risk_logs(events, duration_seconds):
    duration = max(float(duration_seconds or 0), 1.0)
    logs = []
    rolling_score = 0.0
    max_second = int(duration)

    for second in range(0, max_second + 1):
        active_events = [
            event for event in events
            if event.start_time_seconds <= second <= event.end_time_seconds
        ]
        if active_events:
            active_pressure = sum(contribution_for_event(event) for event in active_events) / 8.0
            overlap_bonus = 0.0
            if len(active_events) == 2:
                overlap_bonus = 5.0
            elif len(active_events) >= 3:
                overlap_bonus = 10.0
            rolling_score = clamp(rolling_score + active_pressure + overlap_bonus)
        else:
            rolling_score = clamp(rolling_score - 4.0)

        category_totals = {category: 0.0 for category in CATEGORIES}
        for event in active_events:
            category = EVENT_CATEGORY.get(event.event_type, "visibility")
            category_totals[category] += contribution_for_event(event)

        logs.append(
            RiskState(
                timestamp_seconds=float(second),
                risk_score=round(rolling_score, 2),
                estimated_safety_score=round(clamp(100 - rolling_score), 2),
                alert_level=alert_level_for_score(rolling_score),
                category_contributions={k: round(v, 2) for k, v in category_totals.items()},
                active_event_types=[event.event_type for event in active_events],
            )
        )

    return logs


def fatigue_trend(events, duration_seconds):
    fatigue_events = [event for event in events if event.event_type in {"drowsiness", "eye_closure", "yawn"}]
    duration = float(duration_seconds or 0)
    if duration < 30 or not fatigue_events:
        return "insufficient_data"

    midpoint = duration / 2.0
    early_score = 0.0
    late_score = 0.0
    for event in fatigue_events:
        value = contribution_for_event(event)
        if event.start_time_seconds < midpoint:
            early_score += value
        else:
            late_score += value

    if early_score == 0:
        return "increasing" if late_score >= 5 else "stable"

    difference = late_score - early_score
    threshold = max(early_score * 0.2, 3.0)
    if difference >= threshold:
        return "increasing"
    if difference <= -threshold:
        return "decreasing"
    return "stable"


def summarize_session(session_id, events, duration_seconds):
    category_totals = categorize_contributions(events)
    percentages = percentage_breakdown(category_totals)
    logs = build_risk_logs(events, duration_seconds)
    final_risk = max((log.risk_score for log in logs), default=0.0)

    event_penalty = sum(contribution_for_event(event) / 6.0 for event in events)
    high_risk_seconds = sum(1 for log in logs if log.risk_score >= 65)
    danger_seconds = sum(1 for log in logs if log.risk_score >= 85)
    risk_time_penalty = high_risk_seconds * 0.15 + danger_seconds * 0.30
    final_safety = clamp(round(100 - event_penalty - risk_time_penalty))

    return SessionAnalysisSummary(
        session_id=session_id,
        final_risk_score=round(final_risk, 2),
        final_safety_score=final_safety,
        alert_level=alert_level_for_score(final_risk),
        fatigue_trend=fatigue_trend(events, duration_seconds),
        category_breakdown=category_totals,
        contribution_percentages=percentages,
        event_count=len(events),
    ), logs
