from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceBehaviorSignal:
    active: bool
    confidence: float
    value: float | None
    threshold: float


class FaceBehaviorSignalExtractor:
    """Convert Face Mesh geometry into independent, model-neutral signals."""

    def __init__(self, app):
        self.mar_threshold = float(app.config.get("MAR_THRESHOLD", 0.60))
        self.horizontal_threshold = float(
            app.config.get("FACE_LOOKING_AWAY_OFFSET_THRESHOLD", 0.07)
        )
        self.yaw_threshold_degrees = float(
            app.config.get("FACE_LOOKING_AWAY_YAW_THRESHOLD_DEGREES", 20.0)
        )
        self.yaw_release_degrees = float(
            app.config.get(
                "FACE_LOOKING_AWAY_YAW_RELEASE_DEGREES",
                max(self.yaw_threshold_degrees - 5.0, 0.0),
            )
        )
        self.down_threshold = float(app.config.get("FACE_HEAD_DOWN_OFFSET_THRESHOLD", 0.08))
        self.calibration_samples = max(
            int(app.config.get("FACE_BEHAVIOR_CALIBRATION_SAMPLES", 20)), 5
        )
        self._neutral_horizontal: deque[float] = deque(maxlen=self.calibration_samples)
        self._neutral_vertical: deque[float] = deque(maxlen=self.calibration_samples)
        self._neutral_yaw: deque[float] = deque(maxlen=self.calibration_samples)
        self._looking_away_active = False

    @property
    def calibrated(self) -> bool:
        return (
            len(self._neutral_horizontal) >= self.calibration_samples
            and len(self._neutral_vertical) >= self.calibration_samples
            and (
                len(self._neutral_yaw) >= self.calibration_samples
                or len(self._neutral_yaw) == 0
            )
        )

    @property
    def calibration_progress(self) -> float:
        count = min(len(self._neutral_horizontal), len(self._neutral_vertical))
        return min(float(count) / float(self.calibration_samples), 1.0)

    @staticmethod
    def _confidence(value: float, threshold: float) -> float:
        if threshold <= 0.0:
            return 1.0
        ratio = max(value, 0.0) / threshold
        return float(np.clip(0.5 + 0.5 * max(ratio - 1.0, 0.0), 0.0, 1.0))

    def observe(self, geometry: dict | None) -> dict[str, object]:
        if not geometry:
            return self._empty()

        mar = float(geometry.get("mar", 0.0) or 0.0)
        horizontal = float(geometry.get("horizontal_offset", 0.0) or 0.0)
        vertical = float(geometry.get("vertical_offset", 0.0) or 0.0)
        has_head_pose = geometry.get("yaw_degrees") is not None
        yaw = float(geometry.get("yaw_degrees", 0.0) or 0.0)

        if not self.calibrated:
            # Calibration estimates the driver's forward-facing neutral pose.
            # Yawning remains observable because MAR does not use this baseline.
            self._neutral_horizontal.append(horizontal)
            self._neutral_vertical.append(vertical)
            if has_head_pose:
                self._neutral_yaw.append(yaw)

        neutral_horizontal = (
            float(np.median(self._neutral_horizontal)) if self._neutral_horizontal else 0.0
        )
        neutral_vertical = (
            float(np.median(self._neutral_vertical)) if self._neutral_vertical else 0.0
        )
        neutral_yaw = float(np.median(self._neutral_yaw)) if self._neutral_yaw else 0.0
        horizontal_delta = horizontal - neutral_horizontal
        vertical_delta = vertical - neutral_vertical
        yaw_delta = yaw - neutral_yaw

        yawning = mar >= self.mar_threshold
        if self.calibrated:
            if has_head_pose:
                # Hysteresis keeps a held head turn active when pose estimates
                # wobble around the boundary. Temporal confirmation is handled
                # by DriverSafetyEventManager after this signal is produced.
                if self._looking_away_active:
                    self._looking_away_active = abs(yaw_delta) > self.yaw_release_degrees
                else:
                    self._looking_away_active = abs(yaw_delta) >= self.yaw_threshold_degrees
                looking_away = self._looking_away_active
                looking_value = yaw_delta
                looking_threshold = self.yaw_threshold_degrees
            else:
                # Compatibility for older callers/tests that only provide the
                # normalized landmark offset.
                looking_away = abs(horizontal_delta) >= self.horizontal_threshold
                looking_value = horizontal_delta
                looking_threshold = self.horizontal_threshold
        else:
            self._looking_away_active = False
            looking_away = False
            looking_value = yaw_delta if has_head_pose else horizontal_delta
            looking_threshold = self.yaw_threshold_degrees if has_head_pose else self.horizontal_threshold
        head_down = self.calibrated and vertical_delta >= self.down_threshold

        return {
            "calibrated": self.calibrated,
            "calibration_progress": self.calibration_progress,
            "neutral_horizontal": neutral_horizontal,
            "neutral_vertical": neutral_vertical,
            "horizontal_delta": horizontal_delta,
            "vertical_delta": vertical_delta,
            "neutral_yaw_degrees": neutral_yaw,
            "yaw_delta_degrees": yaw_delta,
            "yawning": FaceBehaviorSignal(
                active=yawning,
                confidence=self._confidence(mar, self.mar_threshold) if yawning else 0.0,
                value=mar,
                threshold=self.mar_threshold,
            ),
            "looking_away": FaceBehaviorSignal(
                active=looking_away,
                confidence=(
                    self._confidence(abs(looking_value), looking_threshold)
                    if looking_away
                    else 0.0
                ),
                value=looking_value,
                threshold=looking_threshold,
            ),
            "head_down": FaceBehaviorSignal(
                active=head_down,
                confidence=(
                    self._confidence(vertical_delta, self.down_threshold) if head_down else 0.0
                ),
                value=vertical_delta,
                threshold=self.down_threshold,
            ),
        }

    def _empty(self) -> dict[str, object]:
        return {
            "calibrated": self.calibrated,
            "calibration_progress": self.calibration_progress,
            "neutral_horizontal": None,
            "neutral_vertical": None,
            "horizontal_delta": None,
            "vertical_delta": None,
            "neutral_yaw_degrees": None,
            "yaw_delta_degrees": None,
            "yawning": FaceBehaviorSignal(False, 0.0, None, self.mar_threshold),
            "looking_away": FaceBehaviorSignal(False, 0.0, None, self.horizontal_threshold),
            "head_down": FaceBehaviorSignal(False, 0.0, None, self.down_threshold),
        }
