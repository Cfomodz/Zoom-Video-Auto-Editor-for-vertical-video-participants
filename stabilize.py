"""
Motion smoothing for user segments (reduces hand shake, preserves intentional movement).
Uses vidstab with a small smoothing window so we smooth high-frequency jitter only.
"""
from pathlib import Path

from config import SMOOTHING_WINDOW, BORDER_TYPE, BORDER_SIZE


def stabilize_video(input_path: Path, output_path: Path, show_progress: bool = True):
    """
    Run vidstab on a video file.
    Uses smoothing_window from config (smaller = preserve more intentional motion).
    """
    try:
        from vidstab import VidStab
    except ImportError:
        raise RuntimeError("Install vidstab: pip install vidstab[cv2]")
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
