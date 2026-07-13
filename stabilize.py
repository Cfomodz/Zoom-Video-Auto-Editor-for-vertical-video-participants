"""
Motion smoothing for user segments (reduces hand shake, preserves intentional movement).
Uses vidstab with a small smoothing window so we smooth high-frequency jitter only.
"""
from pathlib import Path

import cv2

from config import SMOOTHING_WINDOW, BORDER_TYPE, BORDER_SIZE


def _make_destroy_all_windows_headless_safe():
    """
    vidstab calls cv2.destroyAllWindows() unconditionally after applying
    transforms, which raises cv2.error on headless OpenCV builds (no highgui) —
    and it raises before the output writer is released, truncating the file.
    Wrap the call so headless builds treat it as a no-op.
    """
    if getattr(cv2.destroyAllWindows, "_headless_safe", False):
        return
    original = cv2.destroyAllWindows

    def safe_destroy_all_windows():
        try:
            original()
        except cv2.error:
            pass  # headless build: no windows exist, nothing to destroy

    safe_destroy_all_windows._headless_safe = True
    cv2.destroyAllWindows = safe_destroy_all_windows


def stabilize_video(input_path: Path, output_path: Path, show_progress: bool = True):
    """
    Run vidstab on a video file.
    Uses smoothing_window from config (smaller = preserve more intentional motion).
    """
    try:
        from vidstab import VidStab
    except (ImportError, AttributeError) as exc:
        # AttributeError: imutils references cv2.BRISK_create at import time,
        # which OpenCV 5 removed from the main module.
        raise RuntimeError(
            "vidstab import failed. Install compatible versions: "
            "pip install 'opencv-contrib-python-headless>=4.8,<5' vidstab "
            f"(error: {exc})"
        ) from exc
    _make_destroy_all_windows_headless_safe()
    stabilizer = VidStab()
    # mp4v for .mp4 compatibility
    stabilizer.stabilize(
        input_path=str(input_path),
        output_path=str(output_path),
        smoothing_window=SMOOTHING_WINDOW,
        border_type=BORDER_TYPE,
        border_size=BORDER_SIZE,
        show_progress=show_progress,
        output_fourcc="mp4v",
    )
