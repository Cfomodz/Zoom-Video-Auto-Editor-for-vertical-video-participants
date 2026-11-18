"""
Frame writer for intermediate and output video files.

A pillarboxed segment is encoded several times on its way through the
pipeline (crop -> stabilize -> blur-fill -> timeline -> final mux). With
OpenCV's mp4v encoder each generation costs quality: on real 1080p webcam
footage PSNR drops from ~40 dB after one generation to ~30 dB after four,
which shows as blockiness. When ffmpeg is available, frames are piped to
libx264 at near-lossless settings instead; otherwise OpenCV's mp4v is used.

Intermediates use 4:4:4 chroma: no chroma loss between generations, and odd
frame widths (a crop can be any width) need no padding. Files a person will
play directly (`final=True`) use 4:2:0, which every player supports.
"""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int], final: bool = False):
        self.path = Path(path)
        self.size = (int(size[0]), int(size[1]))
        w, h = self.size
        self._proc = None
        self._cv = None
        if shutil.which("ffmpeg") is not None and (not final or (w % 2 == 0 and h % 2 == 0)):
            self._proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
                    "-i", "-",
                    "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "18" if final else "12",
                    "-pix_fmt", "yuv420p" if final else "yuv444p",
                    *(["-movflags", "+faststart"] if final else []),
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
            )
        else:
            self._cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)

    def write(self, frame: np.ndarray) -> None:
        if self._cv is not None:
            self._cv.write(frame)
            return
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, writer expects {self.size}")
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def release(self) -> None:
        if self._cv is not None:
            self._cv.release()
            return
        if self._proc is not None:
            self._proc.stdin.close()
            rc = self._proc.wait()
            self._proc = None
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path} (exit {rc})")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
