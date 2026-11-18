"""
ffprobe/ffmpeg helpers for timing: stream start offsets and variable frame rate.

The pipeline addresses video by frame index (OpenCV) and converts to time as
index / fps. That is only correct for constant-frame-rate input. Call-recording
apps (Zoom in particular) commonly write variable-frame-rate MP4s, dropping
frames while the picture is static; frame-index timing then drifts against the
audio by hundreds of milliseconds. `ensure_cfr` detects that and converts the
input to a constant-frame-rate working copy once, before any other step.
"""
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _ffprobe_json(path: Path, *args: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", *args, "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout or "{}")


def video_start_offset(path: Path) -> float:
    """
    Seconds between the start of ffmpeg's input timeline and the first video
    frame. ffmpeg shifts every input so its earliest stream starts at 0; if the
    audio starts before the video (common with recorders), OpenCV's frame 0 sits
    this far into the audio. 0.0 when ffprobe is unavailable or fields are missing.
    """
    if shutil.which("ffprobe") is None:
        return 0.0
    info = _ffprobe_json(path, "-show_entries", "format=start_time:stream=codec_type,start_time")
    try:
        fmt_start = float(info["format"]["start_time"])
        video = next(s for s in info.get("streams", []) if s.get("codec_type") == "video")
        return max(0.0, float(video["start_time"]) - fmt_start)
    except (KeyError, StopIteration, TypeError, ValueError):
        return 0.0


def frame_timing(path: Path) -> tuple[float, float] | None:
    """
    Return (nominal_fps, max_drift_seconds) for the first video stream, where
    max_drift is the worst gap between a frame's real timestamp and where
    index / fps would put it. None if there are too few frames to judge.
    """
    info = _ffprobe_json(
        path, "-select_streams", "v:0", "-show_entries", "packet=pts_time:stream=r_frame_rate"
    )
    pts = np.array(
        sorted(float(p["pts_time"]) for p in info.get("packets", []) if p.get("pts_time") not in (None, "N/A"))
    )
    if len(pts) < 3:
        return None
    dts = np.diff(pts)
    median_dt = float(np.median(dts))
    if median_dt <= 0:
        return None
    nominal = 1.0 / median_dt
    try:
        r = float(Fraction(info["streams"][0]["r_frame_rate"]))
        if r > 0 and abs(r - nominal) / nominal < 0.01:
            nominal = r  # prefer the exact declared rate (e.g. 30000/1001) when it matches
    except (KeyError, IndexError, ValueError, ZeroDivisionError):
        pass
    expected = pts[0] + np.arange(len(pts)) / nominal
    return nominal, float(np.max(np.abs(pts - expected)))


def is_variable_frame_rate(path: Path) -> bool:
    """True if index / fps misplaces some frame by more than half a frame."""
    timing = frame_timing(path)
    if timing is None:
        return False
    fps, drift = timing
    return drift > 0.5 / fps


def ensure_cfr(input_path: Path, work_dir: Path, cache_key: str, resume: bool = True) -> Path:
    """
    Return a path the pipeline can address by frame index: the input itself
    when it is constant-frame-rate, otherwise a CFR copy in work_dir (made once,
    reused on resume). Audio is copied unchanged into the .mkv working copy.
    """
    if not ffmpeg_available():
        print("  ffmpeg/ffprobe not found; assuming constant frame rate (audio may drift on VFR input).")
        return input_path
    timing = frame_timing(input_path)
    if timing is None:
        return input_path
    fps, drift = timing
    if drift <= 0.5 / fps:
        return input_path
    out = work_dir / f"{cache_key}_cfr.mkv"
    if resume and out.exists() and out.stat().st_size > 0:
        print(f"  Reusing constant-frame-rate copy: {out.name}")
        return out
    print(
        f"  Input is variable frame rate (frames up to {drift * 1000:.0f} ms off a {fps:.3f} fps grid); "
        "converting to constant frame rate first…"
    )
    tmp = out.with_name(out.stem + ".part.mkv")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(input_path),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-fps_mode", "cfr", "-r", f"{fps:.6f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(out)
    return out
