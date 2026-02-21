<div align="center">
  
# Video processing for conference recordings
![GitHub License](https://img.shields.io/github/license/Cfomodz/Zoom-Video-Auto-Editor-for-vertical-video-participants)
![GitHub Sponsors](https://img.shields.io/github/sponsors/Cfomodz)
![Discord](https://img.shields.io/discord/425182625032962049)

<img src="https://github.com/user-attachments/assets/2369c7b0-c391-4a7c-92d3-1029499760c2" alt="video conference" width="400"/>

</div>

---

**Process a recorded Zoom, Google Meet, or similar call where:**

- **Some participants** have horizontal, stationary footage (e.g. webcam) — left unchanged.
- **Others** have vertical and/or unsteady footage (e.g. phone) — crop pillarboxing, smooth shake, and place in a 1920×1080 frame with a blurred background fill.

The pipeline detects scene cuts, classifies each segment as pillarboxed (vertical/unsteady) or full-width (horizontal/stationary), then processes only the pillarboxed segments: crop to content, stabilize, and blur-fill into a standard horizontal frame.

## Requirements

- Python 3.10+
- FFmpeg on `PATH` (for PySceneDetect)
- OpenCV (installed via `requirements.txt`)

## Install

```bash
cd video_processing
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Configuration

1. **Paths** — In `config.py`, set `INPUT_VIDEO` to your recording (e.g. `THIS_DIR / "input.mp4"`). `OUTPUT_DIR` and `OUTPUT_VIDEO` control where results are written.

2. **Tuning** — Same file: scene detection thresholds, pillarbox detection (`BLACK_THRESHOLD`, `PILLAR_EDGE_WINDOW`, etc.), blur strength, and motion-smoothing window. See comments in `config.py`.

## Usage

```bash
# Full pipeline (crop → stabilize → blur-fill); writes processed video to config OUTPUT_VIDEO
.venv/bin/python pipeline.py --input /path/to/recording.mp4 --output /path/to/processed.mp4

# Full timeline: same length as input, with pillarboxed segments replaced by processed versions
.venv/bin/python pipeline.py --input /path/to/recording.mp4 --output /path/to/processed.mp4 --full-timeline

# Skip stabilization (faster)
.venv/bin/python pipeline.py --input /path/to/recording.mp4 --output /path/to/out.mp4 --skip-stabilize

# Force full recompute (ignore cached segments and scene cache)
.venv/bin/python pipeline.py --input /path/to/recording.mp4 --no-resume
```

If you omit `--input` / `--output`, paths from `config.py` are used.

**Inspect scenes** (list segment boundaries and pillarboxed vs full-width):

```bash
.venv/bin/python inspect_scenes.py /path/to/recording.mp4
.venv/bin/python inspect_scenes.py /path/to/recording.mp4 --refresh   # recompute scene cache
```

**Debug pillarbox detection / blur-fill** on a single frame:

```bash
.venv/bin/python debug_blur.py /path/to/recording.mp4 --segment 0 --save --cropped
.venv/bin/python debug_blur.py /path/to/recording.mp4 --dump   # write per-column CSV for analysis
```

## Output

- **Default** (no `--full-timeline`): one video containing only the processed (pillarboxed) segments, in order.
- **With `--full-timeline`**: one video with the same duration as the input; full-width segments are copied from the original, pillarboxed segments are replaced by the processed versions.

Processed segments are written under `output/` (temp crops, stabilized clips, and final blur-filled clips). The pipeline is resumable: re-run the same command to reuse existing temp files and only recompute what’s missing.

## License

MIT — see [LICENSE](LICENSE).
