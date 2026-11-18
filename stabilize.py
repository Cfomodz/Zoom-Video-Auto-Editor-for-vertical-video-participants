"""
Motion smoothing for pillarboxed (phone) segments: removes hand shake while
keeping intentional movement, without letting fill pixels into the frame.

1. Motion: Shi-Tomasi corners tracked with Lucas-Kanade optical flow; a
   RANSAC partial affine fit (translation + rotation) per frame pair.
2. Path: the cumulative camera path is smoothed with a *centered* moving
   average of SMOOTHING_WINDOW frames (a trailing average lags intentional
   pans and over-corrects them).
3. Warp: each frame is moved onto the smoothed path, rotating about its
   center. Moving a frame uncovers its edges. Filling them shows up as smeared
   streaks ("replicate") or black slivers ("black") right at the content edge
   — it looks like a bad crop. Instead the whole segment is zoomed in just
   enough to hide the largest correction, at most STABILIZE_MAX_ZOOM; any
   correction that would need more is scaled back to fit.

Two passes over the file (motion, then warp), so memory stays flat for long
segments. Frame count and frame size are preserved.
"""
import math
from pathlib import Path

import cv2
import numpy as np

from config import SMOOTHING_WINDOW, STABILIZE_MAX_ZOOM
from video_io import VideoWriter

# Motion is estimated at full resolution up to this size. At half resolution
# the per-frame estimate was ~1 px noisier, which on a steady selfie clip
# *doubled* its jitter (the noise goes straight into the corrections).
_ANALYSIS_MAX_DIM = 1080
_MIN_POINTS = 8
_RANSAC_PX = 3.0  # inlier threshold, full-resolution pixels
_MAX_SHIFT_RATIO = 0.06  # per-frame shift beyond 6% of the frame's long side is not shake
_MAX_ROTATION = math.radians(3)  # per-frame rotation beyond this is not shake


def _pair_motion(prev: np.ndarray, gray: np.ndarray, scale: float) -> tuple[float, float, float]:
    """
    Camera motion between two (possibly downscaled) gray frames as
    (dx, dy, da): the shift of the frame center in full-resolution pixels and
    the rotation in radians. Zeros when there is too little to track, or when
    the motion is far too large to be hand shake (a cut, a whip pan): such a
    step would enter the integrated camera path and the smoother would spread
    it over the whole window — worse than leaving that frame uncorrected.
    """
    pts = cv2.goodFeaturesToTrack(
        prev, maxCorners=200, qualityLevel=0.01, minDistance=max(5, int(30 * scale))
    )
    if pts is None or len(pts) < _MIN_POINTS:
        return (0.0, 0.0, 0.0)
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
    tracked = status.reshape(-1) == 1
    if tracked.sum() < _MIN_POINTS:
        return (0.0, 0.0, 0.0)
    m, inliers = cv2.estimateAffinePartial2D(
        pts[tracked], nxt[tracked], method=cv2.RANSAC, ransacReprojThreshold=_RANSAC_PX * scale
    )
    if m is None or inliers is None or inliers.sum() < _MIN_POINTS:
        return (0.0, 0.0, 0.0)
    # m maps p -> A p + t with rotation/scale about the image origin (top-left),
    # so t alone mixes rotation/scale noise (times the distance from the corner)
    # into the shift. Use the motion of the frame center instead: that is the
    # translation that pairs with the rotation about the center used in warping.
    h, w = prev.shape[:2]
    center = np.array([w / 2, h / 2])
    shift = m[:, :2] @ center + m[:, 2] - center
    dx, dy, da = shift[0] / scale, shift[1] / scale, math.atan2(m[1, 0], m[0, 0])
    if max(abs(dx), abs(dy)) > _MAX_SHIFT_RATIO * max(w, h) / scale or abs(da) > _MAX_ROTATION:
        return (0.0, 0.0, 0.0)
    return (dx, dy, da)


def estimate_motion(video_path: Path) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    Per-frame motion relative to the previous frame as rows of (dx, dy, da),
    in full-resolution pixels and radians (row 0 is zeros). Also returns
    (width, height) and fps.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, _ANALYSIS_MAX_DIM / max(w, h, 1))
    deltas, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        delta = (0.0, 0.0, 0.0)
        if prev is not None:
            delta = _pair_motion(prev, gray, scale)
        deltas.append(delta)
        prev = gray
    cap.release()
    return np.array(deltas, dtype=np.float64).reshape(-1, 3), (w, h), fps


def smooth_path(path: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along axis 0 (edges padded with the end values)."""
    n = len(path)
    if n == 0 or window <= 1:
        return path.copy()
    window = min(window, n)
    half = window // 2
    padded = np.pad(path, ((half, window - 1 - half), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.column_stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(path.shape[1])])


def _uncovered_margins(corr: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal / vertical edge margin each frame's correction uncovers (pixels)."""
    dx, dy, da = corr[:, 0], corr[:, 1], corr[:, 2]
    s, c = np.abs(np.sin(da)), 1 - np.cos(da)
    mx = np.abs(dx) + s * h / 2 + c * w / 2
    my = np.abs(dy) + s * w / 2 + c * h / 2
    return mx, my


def correction_limits(w: int, h: int, max_zoom: float, rotation_needed: float = math.inf) -> np.ndarray:
    """
    Largest |dx|, |dy|, |da| a correction may have so that zooming by
    max_zoom still hides the uncovered edges. Rotation gets what it needs, up
    to a third of the margin; translation gets the rest.
    """
    allow_x = w * (1 - 1 / max_zoom) / 2
    allow_y = h * (1 - 1 / max_zoom) / 2
    a_max = min(math.asin(min(1.0, allow_x / 3 / (h / 2))), math.asin(min(1.0, allow_y / 3 / (w / 2))))
    a_max = min(a_max, rotation_needed)
    s, c = math.sin(a_max), 1 - math.cos(a_max)
    return np.array([allow_x - (s * h / 2 + c * w / 2), allow_y - (s * w / 2 + c * h / 2), a_max])


def plan_corrections(
    deltas: np.ndarray, w: int, h: int, window: int, max_zoom: float
) -> tuple[np.ndarray, float]:
    """
    Corrections (dx, dy, da) per frame and the zoom that hides every uncovered
    edge.

    The smoothed camera path is kept inside a corridor around the real path
    (limits from correction_limits): smooth, clip to the corridor, re-smooth,
    until it settles. Big intentional moves are then followed smoothly instead
    of each frame's correction being cut back on its own, which would put
    jitter right back in.
    """
    if len(deltas) == 0:
        return deltas.copy(), 1.0
    trajectory = np.cumsum(deltas, axis=0)
    path = smooth_path(trajectory, window)
    lim = correction_limits(w, h, max_zoom, rotation_needed=float(np.abs(path - trajectory)[:, 2].max()))
    for _ in range(50):
        constrained = trajectory + np.clip(path - trajectory, -lim, lim)
        new_path = smooth_path(constrained, window)
        if np.allclose(new_path, path, atol=0.05):
            path = new_path
            break
        path = new_path
    corr = np.clip(path - trajectory, -lim, lim)
    mx, my = _uncovered_margins(corr, w, h)
    need = max(w / max(w - 2 * mx.max(), 1), h / max(h - 2 * my.max(), 1))
    zoom = float(min(max_zoom, need * 1.002 + 1e-6)) if need > 1.0 else 1.0
    return corr, zoom


def stabilize_video(input_path: Path, output_path: Path, show_progress: bool = True):
    """Stabilize input_path into output_path (same size, same frame count)."""
    deltas, (w, h), fps = estimate_motion(input_path)
    corr, zoom = plan_corrections(deltas, w, h, SMOOTHING_WINDOW, STABILIZE_MAX_ZOOM)
    if show_progress:
        print(f"    stabilize: {len(corr)} frames, zoom {zoom:.3f}")
    center = (w / 2, h / 2)
    cap = cv2.VideoCapture(str(input_path))
    out = VideoWriter(output_path, fps, (w, h))
    try:
        for dx, dy, da in corr:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            m = cv2.getRotationMatrix2D(center, -math.degrees(da), zoom)
            m[0, 2] += zoom * dx
            m[1, 2] += zoom * dy
            # Replicate only guards sub-pixel rounding; the zoom keeps fill out of view.
            out.write(cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    finally:
        cap.release()
        out.release()
