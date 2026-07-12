"""Background-thread webcam capture with a latest-frame-only slot.

The capture thread reads continuously and keeps only the most recent frame;
stale frames are dropped so consumer latency stays bounded regardless of how
fast the downstream pipeline reads. ``cv2.VideoCapture.read`` allocates a fresh
array per call, so the reference returned to the consumer is never mutated by
the next capture.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class WebcamCapture:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._new = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {self.index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="webcam", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame
                self._new = True

    def get_latest(self) -> tuple[Optional[np.ndarray], bool]:
        """Return ``(frame_bgr, is_new)``.

        ``is_new`` is True only on the first call after a fresh frame arrived,
        so the consumer can skip redundant tracking work when re-using a frame.
        """
        with self._lock:
            frame = self._latest
            new = self._new
            self._new = False
            return frame, new

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        self._cap = None
