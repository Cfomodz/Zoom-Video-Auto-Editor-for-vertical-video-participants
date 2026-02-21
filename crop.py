"""
Step 1 for user segments: crop to content only (true vertical video, no black bars).
Uses a stable content bounds (from sampling a few frames) so the crop is consistent.
"""
import cv2
from pathlib import Path

from config import BLACK_THRESHOLD
from scene_utils import pillarbox_content_bounds


def get_segment_content_bounds(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    num_samples: int = 5,
) -> tuple[int, int] | None:
    """
    Sample frames in the segment and return content bounds (x_left, x_right).
    Uses the intersection across samples: content_left = max(lefts), content_right = min(rights),
    so we never include left or right black even if one frame had wrong bounds.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = end_frame - start_frame
    step = max(1, total // (num_samples + 1))
    lefts, rights = [], []
    for i in range(start_frame, end_frame, step):
        if len(lefts) >= num_samples:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        bounds = pillarbox_content_bounds(frame, black_threshold=BLACK_THRESHOLD)
        if bounds is not None:
            lefts.append(bounds[0])
            rights.append(bounds[1])
    cap.release()
    if not lefts or not rights:
        return None
    # Intersection: narrowest content region so no black bar from any frame gets in
    x_left = max(lefts)
    x_right = min(rights)
    if x_right <= x_left:
        return None
    return (x_left, x_right)


def crop_segment(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    output_path: Path,
    bounds: tuple[int, int],
    fps: float,
):
    """
    Read segment from video_path, crop each frame to [x_left, x_right), write to output_path.
    Output is true vertical video (content only, no black bars). Dimensions: (content_width, height).
    """
    x_left, x_right = bounds
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = x_right - x_left
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_w, h))
    total = end_frame - start_frame
    for _ in range(total):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        cropped = frame[:, x_left:x_right, :]
        out.write(cropped)
    cap.release()
    out.release()
