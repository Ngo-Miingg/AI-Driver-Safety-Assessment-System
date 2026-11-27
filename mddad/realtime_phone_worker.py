from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhoneWorkerResult:
    timestamp_seconds: float
    detections: tuple[dict[str, Any], ...] = ()
    error: str | None = None


class LatestFrameDetectorWorker:
    """Run a slow detector without blocking the camera capture loop."""

    def __init__(self, detector):
        self._detector = detector
        self._queue: queue.Queue[tuple[float, np.ndarray] | None] = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._result: PhoneWorkerResult | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="phone-detector", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, timestamp_seconds: float) -> None:
        item = (float(timestamp_seconds), frame.copy())
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass

    def poll(self) -> PhoneWorkerResult | None:
        with self._result_lock:
            result = self._result
            self._result = None
        return result

    def close(self, timeout: float = 3.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=max(float(timeout), 0.0))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return
            timestamp_seconds, frame = item
            try:
                detections = tuple(self._detector.detect_frame(frame) or ())
                result = PhoneWorkerResult(timestamp_seconds, detections=detections)
            except Exception as exc:
                result = PhoneWorkerResult(timestamp_seconds, error=str(exc))
            with self._result_lock:
                self._result = result
