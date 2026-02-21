"""
Configuration for conference-recording video processing.
Adjust paths and thresholds for your source video (resolution, framerate, layout).
"""
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "output"
INPUT_VIDEO = THIS_DIR / "input.mp4"
OUTPUT_VIDEO = OUTPUT_DIR / "processed.mp4"

# Scene detection (PySceneDetect)
SCENE_DETECTOR = "adaptive"  # "adaptive" | "content" | "threshold"
SCENE_MIN_LEN = 15
SCENE_THRESHOLD = 3.0

# Pillarbox detection: segments with black L/R bars (vertical or letterboxed content)
BLACK_THRESHOLD = 25
PILLAR_MIN_WIDTH_RATIO = 0.05
PILLAR_MIN_CONTENT_RATIO = 0.15
PILLAR_EDGE_WINDOW = 10
PILLAR_SAMPLE_FRAMES = 5

# Blur-fill
BLUR_KERNEL = (99, 99)
BLUR_SCALE = 0.15
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

# Motion smoothing (vidstab) on cropped segments
SMOOTHING_WINDOW = 15
BORDER_TYPE = "replicate"
BORDER_SIZE = 0
