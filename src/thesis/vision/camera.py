"""Frame sources: Intel RealSense D435, a webcam, or files (for testing without a camera)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class FrameSource(Protocol):
    name: str

    def read(self) -> np.ndarray | None:
        """Return the newest BGR frame, or None when the source has ended."""

    def close(self) -> None: ...


class RealSenseSource:
    """Colour stream of an Intel RealSense camera (D435)."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        import pyrealsense2 as rs  # optional dependency, only needed for the live camera

        self.name = f"realsense {width}x{height}@{fps}"
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipe.start(cfg)

    def read(self) -> np.ndarray | None:
        frames = self._pipe.wait_for_frames(timeout_ms=5000)
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()).copy() if color else None

    def close(self) -> None:
        self._pipe.stop()


class WebcamSource:
    """Any camera OpenCV can open by index (fallback when no RealSense is attached)."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.name = f"webcam {index}"
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {index}")

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class FileSource:
    """A single image, a folder of images (playlist, sorted by name) or a video file."""

    def __init__(self, path: str | Path, playlist_seconds: float = 4.0, loop: bool = False):
        self.path = Path(path)
        self.name = f"file {self.path.name}"
        self.playlist_seconds = playlist_seconds
        self.loop = loop
        self._cap = None
        self._images: list[np.ndarray] = []
        if self.path.is_dir():
            files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_EXT)
            self._images = [cv2.imread(str(p)) for p in files]
            self.names = [p.name for p in files]
        elif self.path.suffix.lower() in IMAGE_EXT:
            self._images = [cv2.imread(str(self.path))]
            self.names = [self.path.name]
            self.loop = True  # a single image never "ends"
        else:
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video {self.path}")
        if self._cap is None and not self._images:
            raise RuntimeError(f"No images found at {self.path}")
        self._t0 = time.monotonic()

    def current_index(self) -> int:
        """Index of the playlist image currently shown."""
        i = int((time.monotonic() - self._t0) / self.playlist_seconds)
        return i % len(self._images) if self.loop else i

    def read(self) -> np.ndarray | None:
        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok and self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            return frame if ok else None
        i = self.current_index()
        return self._images[i].copy() if i < len(self._images) else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def open_source(spec: str, cam_cfg: dict) -> FrameSource:
    """Create a frame source from a spec: 'realsense', 'webcam:<n>', or a file/folder path."""
    w, h, fps = cam_cfg.get("width", 1280), cam_cfg.get("height", 720), cam_cfg.get("fps", 30)
    if spec == "realsense":
        return RealSenseSource(w, h, fps)
    if spec.startswith("webcam"):
        idx = int(spec.split(":", 1)[1]) if ":" in spec else 0
        return WebcamSource(idx, w, h)
    return FileSource(spec, cam_cfg.get("playlist_seconds", 4.0))
