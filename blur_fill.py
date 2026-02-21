"""
Replace pillarbox (black bars) with a zoomed, heavily blurred version of the video.
Output is full 1920x1080 with the vertical content centered on a soft blurred background.

Two modes:
- blur_fill_frame: input is full frame with pillarbox; detects content and fills sides.
- blur_fill_cropped_frame: input is already cropped (vertical video, no black); whole frame = content.
"""
import cv2
import numpy as np
from pathlib import Path
from config import (
    BLUR_KERNEL,
    BLUR_SCALE,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    BLACK_THRESHOLD,
    PILLAR_MIN_WIDTH_RATIO,
)
from scene_utils import pillarbox_content_bounds


def blur_fill_cropped_frame(frame: np.ndarray) -> np.ndarray:
    """
    Input frame is already cropped (vertical video, no black bars). Whole frame = content.
    Build a 1920x1080 frame: blurred zoomed version of this frame as background,
    then the sharp content scaled to fit height and centered on top.
    """
    ch, cw = frame.shape[:2]
    # Background: blur this frame (all content), scale to fill output
    small = cv2.resize(frame, None, fx=BLUR_SCALE, fy=BLUR_SCALE, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, BLUR_KERNEL, 0)
    background = cv2.resize(blurred, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    # Content: scale to fit height, center horizontally
    scale = OUTPUT_HEIGHT / ch
    new_cw = int(cw * scale)
    new_ch = OUTPUT_HEIGHT
    content_scaled = cv2.resize(frame, (new_cw, new_ch), interpolation=cv2.INTER_LINEAR)
    x_off = (OUTPUT_WIDTH - new_cw) // 2
    background[0:new_ch, x_off : x_off + new_cw] = content_scaled
    return background


def blur_fill_frame(frame: np.ndarray, content_bounds=None):
    """
    One frame: use content_bounds (x_left, x_right) or detect pillarbox.
    Returns a full OUTPUT_WIDTH x OUTPUT_HEIGHT frame with content centered
    and sides filled by zoomed blurred version of the frame.
    """
    h, w = frame.shape[:2]
    if content_bounds is None:
        content_bounds = pillarbox_content_bounds(frame, black_threshold=BLACK_THRESHOLD)
    if content_bounds is None:
        # No pillarbox: return frame scaled to output (no blur fill)
        return cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    x_left, x_right = content_bounds
    content = frame[:, x_left:x_right, :].copy()
    ch, cw = content.shape[:2]

    # Background: use content only (not the black bars) so the fill is blurred video, not blurred black
    small = cv2.resize(content, None, fx=BLUR_SCALE, fy=BLUR_SCALE, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, BLUR_KERNEL, 0)
    background = cv2.resize(blurred, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # Scale content to fit height (preserve aspect); center horizontally
    scale = OUTPUT_HEIGHT / ch
    new_cw = int(cw * scale)
    new_ch = OUTPUT_HEIGHT
    content_scaled = cv2.resize(content, (new_cw, new_ch), interpolation=cv2.INTER_LINEAR)
    x_off = (OUTPUT_WIDTH - new_cw) // 2
    # Overlay content onto background
    background[0:new_ch, x_off : x_off + new_cw] = content_scaled
    return background


def blur_fill_segment(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    output_path: Path,
    fps: float,
    progress_callback=None,
):
    """
    Read segment from video_path, apply blur-fill per frame, write to output_path.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    total = end_frame - start_frame
    for i in range(total):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        filled = blur_fill_frame(frame)
        out.write(filled)
        if progress_callback and (i + 1) % 30 == 0:
            progress_callback(i + 1, total)
    cap.release()
    out.release()


def blur_fill_segment_from_cropped(
    cropped_video_path: Path,
    output_path: Path,
    fps: float,
    progress_callback=None,
):
    """
    Read from already-cropped video (vertical, no black). Each frame is content only;
    apply blur_fill_cropped_frame to place it in 1920x1080 with blurred background.
    """
    cap = cv2.VideoCapture(str(cropped_video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        filled = blur_fill_cropped_frame(frame)
        out.write(filled)
        if progress_callback and (i + 1) % 30 == 0:
            progress_callback(i + 1, total)
    cap.release()
    out.release()
