# AI Driver Safety Assessment System - Scoring Engine Design

## 1. Purpose

This document locks the rule-based scoring engine design before implementation. It defines event types, severity levels, risk contributions, risk score formula, safety score formula, alert logic, explainable risk output, and fatigue trend forecast.

The design is intentionally practical for Python implementation, realtime updates, SQLite logging, and a solo 15-day graduation project.

## 2. Design Principles

- Use transparent rule-based scoring.
- Avoid complex ML scoring.
- Score temporal events, not isolated frames.
- Smooth short noise and prevent false positives from single-frame detections.
- Keep formulas easy to explain during defense.
- Keep realtime calculation lightweight for RTX 3050 Laptop GPU 6GB.
- Support both webcam realtime and uploaded video analysis.

## 3. Event Types

### 3.1 Eye Closure

Source:

- EAR analysis from MediaPipe eye landmarks.

Confirmed when:

- EAR remains below threshold longer than the minimum event duration.

Meaning:

- Possible drowsiness indicator.

### 3.2 Yawn

Source:

- MAR analysis from MediaPipe mouth landmarks.

Confirmed when:

- MAR remains above threshold long enough to indicate yawning.

Meaning:

- Fatigue indicator.

### 3.3 Phone Usage

Source:

- YOLO11n/s phone detection.

Confirmed when:

- Phone is detected repeatedly or continuously within a temporal window.

Meaning:

- High distraction risk.

### 3.4 Looking Away

Source:

- Head pose analysis from MediaPipe landmarks.

Confirmed when:

- Head direction remains left, right, up, or otherwise non-forward beyond a minimum duration.

Meaning:

- Driver attention is away from the road.

### 3.5 Head Down

Source:

- Head pose analysis.

Confirmed when:

- Head orientation indicates downward attention beyond a minimum duration.

Meaning:

- Strong distraction or drowsiness risk, especially when combined with phone detection or eye closure.

### 3.6 Face Missing

Source:

- MediaPipe fails to detect usable face landmarks.

Confirmed when:

- Face remains missing beyond a tolerance period.

Meaning:

- Analysis confidence is reduced; may indicate occlusion, extreme head turn, poor lighting, or driver out of camera view.

## 4. Severity Levels

Severity is assigned after an event is closed or updated while active.

### 4.1 Low

Meaning:

- Short event or weak confidence.

Typical conditions:

- Event passes minimum duration but ends quickly.
- Detection confidence is acceptable but not strong.
- No overlap with other risk events.

### 4.2 Medium

Meaning:

- Clearly risky event with moderate duration or repetition.

Typical conditions:

- Event lasts longer than low threshold.
- Event repeats within a short period.
- Confidence is stable.

### 4.3 High

Meaning:

- Strong risky behavior likely to affect driving safety.

Typical conditions:

- Event duration is long.
- Event repeats often.
- Event overlaps with another risky event.
- Event type has high inherent risk such as phone usage.

### 4.4 Critical

Meaning:

- Very high-risk behavior requiring highest alert level.

Typical conditions:

- Prolonged eye closure.
- Long phone usage.
- Head down for extended time.
- Multiple high-risk events overlap.
- Sustained face missing during active monitoring.

## 5. Recommended Event Confirmation Durations

These values are locked as initial implementation defaults and can be tuned later.

| Event Type | Candidate Starts When | Confirmed After | Merge Gap |
|---|---:|---:|---:|
| Eye Closure | EAR below threshold | 1.0 second | 0.5 second |
| Yawn | MAR above threshold | 1.2 seconds | 1.0 second |
| Phone Usage | phone detected | 1.0 second or 3 detections in 2 seconds | 1.0 second |
| Looking Away | non-forward head pose | 1.5 seconds | 0.75 second |
| Head Down | downward head pose | 1.2 seconds | 0.75 second |
| Face Missing | no usable face | 2.0 seconds | 1.0 second |

## 6. Severity Duration Rules

| Event Type | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Eye Closure | 1.0-1.9s | 2.0-3.4s | 3.5-5.9s | >= 6.0s |
| Yawn | 1.2-2.4s | 2.5-4.0s | > 4.0s or repeated | repeated with eye closure |
| Phone Usage | 1.0-2.4s | 2.5-5.0s | 5.1-10.0s | > 10.0s |
| Looking Away | 1.5-2.9s | 3.0-5.0s | 5.1-8.0s | > 8.0s |
| Head Down | 1.2-2.4s | 2.5-4.0s | 4.1-7.0s | > 7.0s |
| Face Missing | 2.0-4.0s | 4.1-7.0s | 7.1-12.0s | > 12.0s |

Repeated means at least three events of the same type occur within a 60-second window.

## 7. Risk Contributions

### 7.1 Base Event Weights

| Event Type | Base Weight |
|---|---:|
| Eye Closure | 18 |
| Yawn | 10 |
| Phone Usage | 25 |
| Looking Away | 16 |
| Head Down | 20 |
| Face Missing | 8 |

### 7.2 Severity Multipliers

| Severity | Multiplier |
|---|---:|
| Low | 0.6 |
| Medium | 1.0 |
| High | 1.5 |
| Critical | 2.2 |

### 7.3 Duration Multiplier

Duration multiplier rewards longer events without making scores explode.

```text
duration_multiplier = min(1.0 + event_duration_seconds / 10.0, 2.0)
```

Examples:

- 2-second event: 1.2x
- 5-second event: 1.5x
- 10-second or longer event: 2.0x maximum

### 7.4 Event Risk Contribution

```text
event_contribution = base_weight * severity_multiplier * duration_multiplier
```

Contribution should be rounded to a practical numeric value for storage and display.

## 8. Risk Score Formula

### 8.1 Inputs

- Active events.
- Newly confirmed events.
- Event type.
- Event severity.
- Event duration.
- Event confidence where available.
- Recent risk score.
- Overlap between active events.
- Time since last update.

### 8.2 Rolling Update Logic

The realtime risk score is maintained on a 0-100 scale.

At each update interval:

```text
risk_score = previous_risk_score
risk_score = risk_score - decay_amount
risk_score = risk_score + active_event_risk
risk_score = risk_score + new_event_boost
risk_score = risk_score + overlap_bonus
risk_score = clamp(risk_score, 0, 100)
```

### 8.3 Decay Logic

Risk should decrease when no risky behavior is active.

Default decay:

```text
decay_amount = 4 points per second when no active event exists
decay_amount = 1 point per second when low-risk event exists
decay_amount = 0 points per second when medium or higher event exists
```

Rationale:

- The system should recover after normal behavior resumes.
- Risk should not instantly drop to zero after a dangerous event.

### 8.4 Active Event Risk

At each second, active events add risk pressure:

```text
active_event_risk = sum(base_weight * active_factor for active events) / 10
```

Recommended active factors:

- Low: 0.4
- Medium: 0.7
- High: 1.0
- Critical: 1.5

### 8.5 New Event Boost

When an event is confirmed or closed, add a one-time boost:

```text
new_event_boost = event_contribution / 8
```

This makes the risk score respond to meaningful confirmed events without relying only on active state.

### 8.6 Overlapping Event Handling

Overlap increases risk because combined risks are more dangerous.

Rules:

- If two active risky events overlap, add 5 points per second.
- If three or more active risky events overlap, add 10 points per second.
- If Phone Usage overlaps with Head Down, add an additional 5 points per second.
- If Eye Closure overlaps with Head Down or Face Missing, add an additional 5 points per second.

The final risk score is always clamped between 0 and 100.

### 8.7 Risk Score Interpretation

| Risk Score | Meaning |
|---:|---|
| 0-19 | Low current risk |
| 20-39 | Mild risk |
| 40-64 | Significant risk |
| 65-84 | High risk |
| 85-100 | Severe risk |

## 9. Safety Score Formula

### 9.1 Starting Score

Each session starts with:

```text
safety_score = 100
```

### 9.2 Event Penalties

For every confirmed event:

```text
event_penalty = event_contribution / 6
```

Recommended maximum single-event penalty:

```text
max_single_event_penalty = 18
```

This prevents one event from destroying the whole score unless it is extremely serious or repeated.

### 9.3 Frequency Penalty

Repeated event types receive additional penalty:

```text
frequency_penalty = max(0, event_count_for_type - 2) * type_frequency_weight
```

Recommended type frequency weights:

| Event Type | Frequency Weight |
|---|---:|
| Eye Closure | 3 |
| Yawn | 2 |
| Phone Usage | 5 |
| Looking Away | 3 |
| Head Down | 4 |
| Face Missing | 1 |

### 9.4 Risk Time Penalty

If a session spends significant time above risk thresholds:

```text
risk_time_penalty =
  high_risk_seconds * 0.15
  + danger_risk_seconds * 0.30
```

Where:

- high risk means risk score >= 65.
- danger risk means risk score >= 85.

### 9.5 Final Safety Score

```text
final_safety_score =
  100
  - sum(event_penalties)
  - sum(frequency_penalties)
  - risk_time_penalty
```

Final normalization:

```text
final_safety_score = clamp(round(final_safety_score), 0, 100)
```

### 9.6 Safety Score Interpretation

| Safety Score | Meaning |
|---:|---|
| 90-100 | Excellent |
| 75-89 | Good |
| 60-74 | Moderate |
| 40-59 | Risky |
| 0-39 | Unsafe |

## 10. Alert Logic

Alert level is derived from current rolling risk score and active critical events.

### 10.1 Thresholds

| Alert Level | Risk Score Range | Additional Trigger |
|---|---:|---|
| Normal | 0-19 | no active medium or higher event |
| Caution | 20-39 | active low event or repeated minor events |
| Warning | 40-69 | active medium/high event |
| Danger | 70-100 | active critical event or severe overlap |

### 10.2 Forced Danger Conditions

Set alert level to Danger regardless of current score if:

- Eye Closure is Critical.
- Phone Usage is Critical.
- Head Down is Critical.
- Three or more risky events overlap.
- Risk score is at least 85.

### 10.3 Forced Warning Conditions

Set alert level to at least Warning if:

- Phone Usage is High.
- Eye Closure is High.
- Head Down is High.
- Looking Away is High.
- Face Missing lasts longer than 7 seconds.

### 10.4 Alert Priority

When multiple rules match:

```text
Danger > Warning > Caution > Normal
```

## 11. Explainable Risk Output

### 11.1 Purpose

Explainable risk output shows why a session received its score.

### 11.2 Category Mapping

| Category | Event Types |
|---|---|
| Drowsiness | Eye Closure, Yawn |
| Distraction | Looking Away, Head Down, Face Missing |
| Phone Usage | Phone Usage |

### 11.3 Contribution Calculation

For each event:

```text
category_total += event_contribution
```

Then:

```text
category_percentage = category_total / all_category_total * 100
```

If total contribution is zero, all categories are 0%.

### 11.4 Example Output

```text
40% Phone Usage
30% Eye Closure
20% Looking Away
10% Yawning
```

### 11.5 Explanation Text Rules

The report should include direct explanations such as:

- Safety score decreased mainly because phone usage was detected for 12 seconds.
- Drowsiness risk increased because prolonged eye closure occurred 3 times.
- Distraction risk increased because the driver looked away for repeated intervals.
- Fatigue trend is increasing because yawning and eye closure became more frequent near the end of the session.

## 12. Fatigue Trend Forecast

### 12.1 Output Labels

- Increasing
- Stable
- Decreasing

### 12.2 Inputs

- Eye closure event count.
- Eye closure total duration.
- Yawn event count.
- Yawn total duration.
- Drowsiness-related risk score trend.
- Session time split into early and late windows.

### 12.3 Window Logic

For sessions at least 60 seconds long:

- Early window: first half of session.
- Late window: second half of session.

For sessions shorter than 60 seconds:

- Use the whole session and mark confidence as low.

### 12.4 Fatigue Indicator Score

Calculate fatigue indicator separately for early and late windows:

```text
fatigue_indicator =
  eye_closure_count * 3
  + eye_closure_total_seconds * 2
  + yawn_count * 2
  + yawn_total_seconds * 1
  + drowsiness_risk_average * 0.2
```

### 12.5 Trend Decision

```text
difference = late_fatigue_indicator - early_fatigue_indicator
```

Rules:

- Increasing if difference >= 20% of early indicator and late indicator is meaningfully above zero.
- Decreasing if difference <= -20% of early indicator.
- Stable otherwise.

If early indicator is zero:

- Increasing if late indicator >= 5.
- Stable if late indicator < 5.

### 12.6 Forecast Confidence

Recommended labels:

- High confidence: session duration >= 5 minutes and at least 3 fatigue-related events.
- Medium confidence: session duration >= 2 minutes and at least 1 fatigue-related event.
- Low confidence: shorter sessions or no fatigue-related events.

### 12.7 Forecast Explanation Examples

Increasing:

- Fatigue indicators increased in the later part of the session, mainly due to more eye closure and yawning events.

Stable:

- Fatigue indicators stayed similar across the session and no strong upward trend was detected.

Decreasing:

- Fatigue indicators were lower in the later part of the session than in the earlier part.

## 13. Storage Requirements for Scoring

Store per session:

- final safety score
- final risk score
- final alert summary
- fatigue trend label
- fatigue forecast explanation
- risk category breakdown

Store per event:

- event type
- severity
- duration
- event contribution
- explanation

Store per risk log:

- timestamp
- risk score
- estimated safety score
- alert level
- optional category contribution summary

## 14. Implementation Notes

- Update rolling risk at a fixed interval, preferably once per second.
- Process frames at a practical rate instead of every camera frame if needed.
- Store risk logs at one-second intervals for charts and reports.
- Clamp all public scores to 0-100.
- Keep thresholds configurable internally, but do not require a settings UI for the first version.
- Use the same scoring engine for realtime and uploaded video analysis.

