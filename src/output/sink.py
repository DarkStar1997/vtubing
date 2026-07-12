"""Output sinks: preview window, file writer, virtual camera.

All sinks accept RGB uint8 frames (h, w, 3) from the renderer and handle
format conversion internally.  The ``sleep_until_next_frame`` method provides
frame-rate pacing.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class Sink(Protocol):
    def send(self, frame: np.ndarray) -> None: ...
    def sleep_until_next_frame(self) -> None: ...
    def close(self) -> None: ...


class WindowPreviewSink:
    def __init__(self, fps: float = 30.0, title: str = "VTubing") -> None:
        self._target_dt = 1.0 / fps
        self._title = title
        self._t_prev = time.monotonic()

    def send(self, frame: np.ndarray) -> None:
        cv2.imshow(self._title, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def sleep_until_next_frame(self) -> None:
        cv2.waitKey(1)
        elapsed = time.monotonic() - self._t_prev
        remaining = self._target_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._t_prev = time.monotonic()

    def close(self) -> None:
        cv2.destroyAllWindows()


class FileSink:
    def __init__(self, path: str | Path, fps: float, width: int, height: int) -> None:
        self._target_dt = 1.0 / fps
        self._t_prev = time.monotonic()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        self._path = str(path)

    def send(self, frame: np.ndarray) -> None:
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def sleep_until_next_frame(self) -> None:
        elapsed = time.monotonic() - self._t_prev
        remaining = self._target_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._t_prev = time.monotonic()

    def close(self) -> None:
        self._writer.release()
        print(f"[sink] wrote {self._path}")


class VirtualCamSink:
    def __init__(self, fps: float, width: int, height: int, backend: str = "obs") -> None:
        import pyvirtualcam
        from pyvirtualcam import Format

        self._cam = pyvirtualcam.Camera(
            width=width, height=height, fps=fps, fmt=Format.RGB,
        )
        print(f"[sink] virtual cam active ({backend}, {width}x{height}@{fps})")

    def send(self, frame: np.ndarray) -> None:
        self._cam.send(frame)

    def sleep_until_next_frame(self) -> None:
        self._cam.sleep_until_next_frame()

    def close(self) -> None:
        self._cam.close()


def create_sink(cfg: dict) -> Sink:
    out_cfg = cfg.get("output", {})
    sink_type = out_cfg.get("sink", "preview")
    fps = out_cfg.get("fps", 30.0)
    width = out_cfg.get("width", 1280)
    height = out_cfg.get("height", 720)

    if sink_type == "preview":
        return WindowPreviewSink(fps=fps)
    elif sink_type == "file":
        path = out_cfg.get("file_path", "output/session.mp4")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return FileSink(path, fps, width, height)
    elif sink_type == "vcam":
        backend = out_cfg.get("vcam", {}).get("backend", "obs")
        return VirtualCamSink(fps, width, height, backend)
    else:
        raise ValueError(f"unknown output.sink: {sink_type}")
