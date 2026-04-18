from __future__ import annotations

import time
from collections import deque


class FpsMeter:
    def __init__(self, window_size: int = 30) -> None:
        self._times = deque(maxlen=window_size)

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed
