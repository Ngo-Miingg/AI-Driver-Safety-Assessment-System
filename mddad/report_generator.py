from .analysis_models import ReportData


def _format_percentages(percentages):
    if not percentages:
        return "No risk contribution was detected."
    return "\n".join(
        f"- {category.replace('_', ' ').title()}: {value:.1f}%"
        for category, value in percentages.items()
    )


def _timeline_summary(events, analysis_mode):
    if not events:
        return f"No {analysis_mode} risk events were generated for this session."
    lines = []
    for event in events:
        lines.append(
            f"- {event.event_type.replace('_', ' ').title()} "
            f"from {event.start_time_seconds:.1f}s to {event.end_time_seconds:.1f}s "
            f"({event.severity}, {event.duration_seconds:.1f}s)."
        )
    return "\n".join(lines)


def _recommendations(summary, analysis_mode):
    recommendations = []
    if summary.contribution_percentages.get("phone_usage", 0) > 0:
        recommendations.append("Avoid handling a phone while driving and keep the device out of reach.")
    if summary.contribution_percentages.get("drowsiness", 0) > 0:
        recommendations.append("Take a rest break when yawning or eye closure appears during a session.")
    if summary.contribution_percentages.get("distraction", 0) > 0:
        recommendations.append("Keep attention forward and minimize off-road glances.")
    if summary.contribution_percentages.get("visibility", 0) > 0:
        recommendations.append("Improve camera position and lighting so the driver's face remains visible.")
    if not recommendations:
        recommendations.append(f"No {analysis_mode} safety issue was generated for this session.")
    return "\n".join(f"- {item}" for item in recommendations)


def generate_report(session, events, summary, analysis_mode="simulated"):
    duration = session["video_duration_seconds"] or session["duration_seconds"] or 0
    title = f"AI Driving Report - Session {session['id']}"
    if analysis_mode == "simulated":
        mode_text = "deterministic simulated mode"
    elif analysis_mode == "multimodal":
        mode_text = "multimodal video-audio-skeleton behavior model mode"
    elif analysis_mode == "multimodal_live":
        mode_text = "live multimodal video-audio-skeleton behavior model mode"
    elif analysis_mode == "full_ai":
        mode_text = "MediaPipe plus phone-detector full AI mode"
    elif analysis_mode == "realtime_mediapipe":
        mode_text = "realtime MediaPipe webcam mode"
    elif analysis_mode == "realtime_mediapipe_yolo":
        mode_text = "realtime MediaPipe webcam mode with optional low-rate phone detection"
    elif analysis_mode == "driver_safety_v1":
        mode_text = "Driver Safety v1 webcam mode with YOLO face detection and temporal BiLSTM drowsiness inference"
    else:
        mode_text = "MediaPipe heuristic mode"
    report_summary = (
        f"Session {session['id']} for driver {session['driver_name']} was analyzed in {mode_text}. "
        f"Duration: {duration:.1f} seconds. "
        f"Generated events: {summary.event_count}."
    )
    safety_explanation = (
        f"The final safety score is {summary.final_safety_score}/100. "
        f"The final risk score is {summary.final_risk_score}/100 with alert level "
        f"{summary.alert_level}."
    )
    key_risks = _format_percentages(summary.contribution_percentages)
    source_text = (
        "simulated eye closure and yawn event distribution"
        if analysis_mode == "simulated"
        else (
            "live video, audio speech signals, skeleton sequences, and driver-state decisions"
            if analysis_mode == "multimodal_live"
            else (
            "clip-level RGB video, audio speech signals, and skeleton sequences"
            if analysis_mode == "multimodal"
            else (
            "MediaPipe-derived face signals and phone-detector detections"
            if analysis_mode == "full_ai"
            else "realtime MediaPipe face signals and low-rate phone detections"
            if analysis_mode == "realtime_mediapipe_yolo"
            else "YOLO face detection and a 15-frame sequence of face landmarks processed by the temporal drowsiness model"
            if analysis_mode == "driver_safety_v1"
            else "realtime MediaPipe face signals"
            if analysis_mode == "realtime_mediapipe"
            else "MediaPipe-derived EAR, MAR, face visibility, and head-direction candidates"
            )
            )
        )
    )
    fatigue_text = (
        f"Fatigue trend forecast: {summary.fatigue_trend}. "
        f"This forecast is based on {source_text}."
    )
    if analysis_mode == "simulated":
        limitations = (
            "This report uses deterministic simulated events. It does not use MediaPipe, "
            "real drowsiness detection, real phone detection, or realtime webcam monitoring yet."
        )
    elif analysis_mode == "multimodal":
        limitations = (
            "This report uses the new multimodal model contract. The model must be trained on video clips, "
            "audio speech/call signals, and skeleton sequences; legacy manual thresholds are not used as the final decision path."
        )
    elif analysis_mode == "multimodal_live":
        limitations = (
            "This report uses the live multimodal model contract. The model receives realtime video, audio, and skeleton cues "
            "from the browser mic and webcam session, then applies driver-state and temporal event rules."
        )
    elif analysis_mode == "full_ai":
        limitations = (
            "This report uses MediaPipe landmarks and phone-detector results. Head pose remains heuristic, "
            "and realtime webcam monitoring is not implemented yet."
        )
    elif analysis_mode == "realtime_mediapipe":
        limitations = (
            "This report uses realtime MediaPipe landmark heuristics. Realtime phone detection "
            "was disabled or unavailable for this session."
        )
    elif analysis_mode == "realtime_mediapipe_yolo":
        limitations = (
            "This report uses realtime MediaPipe landmark heuristics and optional low-rate phone "
            "detection. The phone detector is sampled at low FPS for performance and is not a full video stream."
        )
    else:
        limitations = (
            "This report uses MediaPipe landmarks and simple heuristics. It does not use phone detection, "
            "production-grade head pose estimation, or realtime webcam monitoring yet."
        )
    return ReportData(
        title=title,
        summary=report_summary,
        safety_score_explanation=safety_explanation,
        key_risk_explanation=key_risks,
        fatigue_forecast_explanation=fatigue_text,
        recommendations=_recommendations(summary, analysis_mode),
        limitations=limitations,
        dominant_risk_breakdown=key_risks,
        timeline_summary=_timeline_summary(events, analysis_mode),
    )
