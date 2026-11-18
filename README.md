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

- Python 3.10–3.13 (tested on each)
- FFmpeg on `PATH` (strongly recommended: H.264 intermediates and output, original audio muxed back in, variable-frame-rate input handled; without it the output is a silent, lower-quality mp4v file)
- OpenCV 5 (headless; installed via `requirements.txt`)

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # or requirements-dev.txt to also get pytest
```

All dependencies are pinned to exact, tested versions. To bump one, change its pin, reinstall in a clean venv, and run `pytest`.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

Unit and synthetic end-to-end tests run offline. To check the pipeline on real footage, build a sample recording from freely licensed Wikimedia Commons videos (real Google Meet webcam segments alternating with real vertical phone video, with audio):

```bash
.venv/bin/python tests/make_real_sample.py      # -> output/real_sample/real_sample.mp4 (+ CREDITS.txt)
.venv/bin/python pipeline.py --input output/real_sample/real_sample.mp4 --output output/real_sample/processed.mp4 --full-timeline
```

## Configuration

1. **Paths** — In `config.py`, set `INPUT_VIDEO` to your recording (e.g. `THIS_DIR / "input.mp4"`). `OUTPUT_DIR` and `OUTPUT_VIDEO` control where results are written.

2. **Tuning** — Same file: scene detection thresholds, pillarbox detection (`BLACK_THRESHOLD`, `PILLAR_EDGE_WINDOW`, etc.), blur strength, motion-smoothing window (`SMOOTHING_WINDOW`) and how far stabilization may zoom in to keep its edges clean (`STABILIZE_MAX_ZOOM`). See comments in `config.py`.

**Stabilization** smooths the phone camera's path (shake removed, deliberate movement kept) and zooms each segment in just enough — at most `STABILIZE_MAX_ZOOM` — that no filled-in edge pixels ever show at the sides of the video.

## Usage

```bash
# Full pipeline (crop → stabilize → blur-fill); writes processed video to config OUTPUT_VIDEO
.venv/bin/python pipeline.py --input /path/to/recording.mp4 --output /path/to/processed.mp4

# Full timeline: same length as input, with pillarboxed segments replaced by processed versions
.venv/bin/python pipeline.py --input /path/to/recording.mp4 --output /path/to/processed.mp4 --full-timeline

# Skip stabilization (faster)
.venv/bin/python pipeline.py --input /path/to/recording.mp4 --output /path/to/out.mp4 --skip-stabilize

# Skip the audio mux step (output is video-only)
.venv/bin/python pipeline.py --input /path/to/recording.mp4 --output /path/to/out.mp4 --no-audio

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

- **Default** (no `--full-timeline`): one video containing only the processed (pillarboxed) segments, in order. Audio for those segments is trimmed from the original and kept in sync.
- **With `--full-timeline`**: one video with the same duration as the input; full-width segments are copied from the original, pillarboxed segments are replaced by the processed versions. The full original audio track is carried over.

When FFmpeg is on `PATH` (and `--no-audio` isn't set), the final file is H.264 + AAC with the original audio, kept in sync in both modes. Otherwise the pipeline falls back to a silent mp4v file and says so.

**Variable frame rate input** (Zoom recordings often are): the pipeline addresses video by frame index, which drifts against the audio on VFR files. It detects this and first converts the input to a constant-frame-rate working copy in `output/temp/` (reused on resume).

Processed segments are written under `output/` (temp crops, stabilized clips, and final blur-filled clips). The pipeline is resumable: re-run the same command to reuse existing temp files and only recompute what’s missing.

## License

MIT — see [LICENSE](LICENSE).
