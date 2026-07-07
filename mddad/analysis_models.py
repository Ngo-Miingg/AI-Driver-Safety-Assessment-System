from dataclasses import dataclass, field


@dataclass(frozen=True)
class FrameAnalysisResult:
    frame_index: int
    timestamp_seconds: float
    detected_states: dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass(frozen=True)
class Event:
    session_id: int
    event_type: str
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    severity: str
    confidence: float
    explanation: str
    risk_contribution: float = 0.0
    evidence_image_path: str | None = None


@dataclass(frozen=True)
class RiskState:
    timestamp_seconds: float
    risk_score: float
    estimated_safety_score: float
    alert_level: str
    category_contributions: dict[str, float]
    active_event_types: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SessionAnalysisSummary:
    session_id: int
    final_risk_score: float
    final_safety_score: float
    alert_level: str
    fatigue_trend: str
    category_breakdown: dict[str, float]
    contribution_percentages: dict[str, float]
    event_count: int


@dataclass(frozen=True)
class ReportData:
    title: str
    summary: str
    safety_score_explanation: str
    key_risk_explanation: str
    fatigue_forecast_explanation: str
    recommendations: str
    limitations: str
    dominant_risk_breakdown: str
    timeline_summary: str
