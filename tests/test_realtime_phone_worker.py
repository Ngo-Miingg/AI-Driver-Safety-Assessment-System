import time

import numpy as np

from mddad.realtime_phone_worker import LatestFrameDetectorWorker


class SlowDetector:
    def detect_frame(self, frame):
        time.sleep(0.15)
        return [{"is_phone": True, "confidence": float(frame[0, 0, 0]) / 255.0}]


def test_phone_worker_does_not_block_camera_thread():
    worker = LatestFrameDetectorWorker(SlowDetector())
    started = time.perf_counter()
    worker.submit(np.full((8, 8, 3), 128, dtype=np.uint8), 1.25)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05

    result = None
    deadline = time.time() + 1.0
    while result is None and time.time() < deadline:
        result = worker.poll()
        time.sleep(0.01)
    worker.close()

    assert result is not None
    assert result.error is None
    assert result.timestamp_seconds == 1.25
    assert result.detections[0]["is_phone"]
