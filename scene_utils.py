"""
Scene detection and classification for conference recordings.
Segments are labeled pillarboxed (vertical/unsteady, black L/R bars) or full_width (horizontal/stationary).

Scene results are cached per video (keyed by path + size + mtime) so resume
and re-runs skip detection/classification when the file is unchanged.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from scenedetect import (
        open_video,
        SceneManager,
        AdaptiveDetector,
        ContentDetector,
        ThresholdDetector,
    )
    from scenedetect.frame_timecode import FrameTimecode
except ImportError:
    open_video = SceneManager = AdaptiveDetector = ContentDetector = ThresholdDetector = None
    FrameTimecode = None

from config import (
    OUTPUT_DIR,
    SCENE_DETECTOR,
    SCENE_MIN_LEN,
    SCENE_THRESHOLD,
    BLACK_THRESHOLD,
    PILLAR_MIN_WIDTH_RATIO,
    PILLAR_MIN_CONTENT_RATIO,
    PILLAR_EDGE_WINDOW,
    PILLAR_SAMPLE_FRAMES,
)


def video_cache_key(video_path: Path) -> str:
    """Stable key for this video file (path + size + mtime). No full-file read."""
    p = video_path.resolve()
    stat = p.stat()
    raw = f"{p!s}\n{stat.st_size}\n{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def scenes_cache_path(video_path: Path) -> Path:
    return OUTPUT_DIR / f"{video_cache_key(video_path)}_scenes.json"


def load_scenes_cache(video_path: Path) -> list[tuple[int, int, str]] | None:
    """
    Load classified scenes from cache if it exists and matches the current video.
    Returns list of (start_frame, end_frame, label) or None if cache miss/invalid.
    """
    path = scenes_cache_path(video_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    key = data.get("video_key")
    if key != video_cache_key(video_path):
        return None
    raw = data.get("classified")
    if not raw or not isinstance(raw, list):
        return None
    out = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        s, e, label = item
        if label in ("user", "tanner"):
            label = "pillarboxed" if label == "user" else "full_width"
        if label not in ("pillarboxed", "full_width"):
            return None
        out.append((int(s), int(e), label))
    return out


def save_scenes_cache(video_path: Path, classified: list[tuple[int, int, str]]) -> None:
    """Save classified scenes to cache for this video."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = scenes_cache_path(video_path)
    data = {
        "video_path": str(video_path.resolve()),
        "video_key": video_cache_key(video_path),
        "classified": [[s, e, label] for s, e, label in classified],
    }
    path.write_text(json.dumps(data, indent=2))


def _detector():
    if SCENE_DETECTOR == "adaptive":
        return AdaptiveDetector(
            adaptive_threshold=SCENE_THRESHOLD,
            min_scene_len=SCENE_MIN_LEN,
        )
    if SCENE_DETECTOR == "content":
        return ContentDetector(
            threshold=SCENE_THRESHOLD,
            min_scene_len=SCENE_MIN_LEN,
        )
    if SCENE_DETECTOR == "threshold":
        return ThresholdDetector(
            threshold=int(SCENE_THRESHOLD),
            min_scene_len=SCENE_MIN_LEN,
        )
    return AdaptiveDetector(adaptive_threshold=SCENE_THRESHOLD, min_scene_len=SCENE_MIN_LEN)


def detect_scenes(video_path: Path):
    """Return list of (start_timecode, end_timecode) for each scene."""
    if SceneManager is None or open_video is None:
        raise RuntimeError("Install scenedetect: pip install scenedetect[opencv]")
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(_detector())
    manager.detect_scenes(video=video)
    scene_list = manager.get_scene_list()
    return scene_list


def is_black_column(
    frame: np.ndarray,
    x: int,
    threshold: int = BLACK_THRESHOLD,
    window: int = PILLAR_EDGE_WINDOW,
) -> bool:
    """
    True if the region around column x is mostly black.
    Averages over ±window columns so 1px lines or encoding artifacts don't flip the result.
    """
    h, w = frame.shape[:2]
    x0 = max(0, x - window)
    x1 = min(w, x + window + 1)
    region = frame[:, x0:x1, :]
    return np.mean(region) < threshold


def pillarbox_content_bounds(frame: np.ndarray, black_threshold: int = BLACK_THRESHOLD):
    """
    If frame has black bars on left/right (pillarbox), return (x_left, x_right) of content.
    Otherwise return None. Uses a horizontal window when testing black so 1px lines don't
    misdefine the content region. Rejects implausibly thin "content" (noise strips).
    """
    h, w = frame.shape[:2]
    min_bar = int(w * PILLAR_MIN_WIDTH_RATIO)
    min_content_width = int(w * PILLAR_MIN_CONTENT_RATIO)
    # Left edge: scan from left until we hit a non-black column (windowed)
    left = 0
    while left < w and is_black_column(frame, left, black_threshold):
        left += 1
    # Refine: the windowed check may stop a few columns early; skip any
    # remaining individually-black columns at the boundary.
    while left < w and np.mean(frame[:, left, :]) < black_threshold:
        left += 1
    if left < min_bar:
        left = 0
    # Right edge: scan from right until we hit a non-black column (windowed)
    right = w - 1
    while right >= 0 and is_black_column(frame, right, black_threshold):
        right -= 1
    # Refine: skip any individually-black columns the window averaged past.
    while right >= 0 and np.mean(frame[:, right, :]) < black_threshold:
        right -= 1
    if right < 0:
        return None
    num_black_right = (w - 1) - right
    if num_black_right < min_bar:
        right = w - 1
    content_width = (right + 1) - left
    if content_width < min_content_width:
        return None
    if left >= min_bar and (w - 1 - right) >= min_bar:
        return (left, right + 1)
    return None


def scene_is_pillarboxed(video_path: Path, start_frame: int, end_frame: int, cap=None) -> bool:
    """True if the majority of sampled frames in the scene are pillarboxed."""
    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
    total = end_frame - start_frame
    step = max(1, total // PILLAR_SAMPLE_FRAMES)
    pillarboxed_count = 0
    sampled = 0
    for i in range(start_frame, end_frame, step):
        if sampled >= PILLAR_SAMPLE_FRAMES:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        sampled += 1
        if pillarbox_content_bounds(frame) is not None:
            pillarboxed_count += 1
    return sampled > 0 and (pillarboxed_count / sampled) >= 0.5


def classify_scenes(video_path: Path, scene_list):
    """
    For each scene, label as 'pillarboxed' or 'full_width'.
    Returns list of (start_frame, end_frame, label).
    """
    cap = cv2.VideoCapture(str(video_path))
    result = []
    for start_tc, end_tc in scene_list:
        start_frame = int(start_tc.get_frames())
        end_frame = int(end_tc.get_frames())
        is_pillarboxed = scene_is_pillarboxed(video_path, start_frame, end_frame, cap=cap)
        result.append((start_frame, end_frame, "pillarboxed" if is_pillarboxed else "full_width"))
    cap.release()
    return result
