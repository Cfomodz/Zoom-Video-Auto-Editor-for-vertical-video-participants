"""Generate a synthetic pillarboxed test image (1920x1080) with black bars on both sides.

The image has:
- Left black bar: columns 0–479 (480px wide, 25% of frame)
- Content region: columns 480–1439 (960px wide, 50% of frame) — colorful noise
- Right black bar: columns 1440–1919 (480px wide, 25% of frame)

Run this script directly to (re)generate the fixture:
    python tests/generate_fixture.py
"""
from pathlib import Path

import cv2
import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# Frame dimensions matching typical Zoom recording
WIDTH = 1920
HEIGHT = 1080

# Black bar width (symmetric)
BAR_WIDTH = 480  # 25% of 1920

CONTENT_LEFT = BAR_WIDTH
CONTENT_RIGHT = WIDTH - BAR_WIDTH


def make_pillarboxed_frame(seed: int = 42) -> np.ndarray:
    """Create a 1920x1080 BGR frame with black bars on both sides and colorful content."""
    rng = np.random.RandomState(seed)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    # Fill the content region with random color noise (values 80-255 to be clearly non-black)
    content = rng.randint(80, 256, (HEIGHT, CONTENT_RIGHT - CONTENT_LEFT, 3), dtype=np.uint8)
    frame[:, CONTENT_LEFT:CONTENT_RIGHT, :] = content
    return frame


def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = make_pillarboxed_frame()
    path = FIXTURE_DIR / "pillarboxed_1920x1080.png"
    cv2.imwrite(str(path), frame)
    print(f"Wrote {path}  ({frame.shape[1]}x{frame.shape[0]})")

    # Also generate a full-width frame (no bars) for negative tests
    rng = np.random.RandomState(99)
    full = rng.randint(80, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    path2 = FIXTURE_DIR / "fullwidth_1920x1080.png"
    cv2.imwrite(str(path2), full)
    print(f"Wrote {path2}  ({full.shape[1]}x{full.shape[0]})")


if __name__ == "__main__":
    main()
