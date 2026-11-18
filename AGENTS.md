# Notes for maintainers and agents

This project is a **standalone repository** intended to be published on its own. It is not part of or tied to any parent directory (e.g. plane-selfhost); treat it as an independent repo.

## What not to commit

Do **not** commit:

- `.venv/` (virtual environment)
- `output/` (processed videos, temp segment files, scene cache JSON, debug images)
- Any source video files (e.g. `input.mp4`, `*.mp4` in the tree)

These are listed in `.gitignore`. If the repo is published, only source code, README, LICENSE, config, and requirements should be tracked.

## Scene cache

Scene detection and classification are cached per video. Cache key = hash of (path, size, mtime). Stored as `output/{video_hash}_scenes.json`. On re-run with the same video, steps 1 and 2 are skipped and the cache is loaded. To force recompute: delete the JSON or run `inspect_scenes.py --refresh`. Cache format: `classified` is a list of `[start_frame, end_frame, label]` with `label` in `["pillarboxed", "full_width"]`. Old caches with `"user"` / `"tanner"` are still loaded and normalized to the new labels.

## Resume behavior

Each pillarboxed segment is written to disk as it’s processed: `output/temp/seg_N_crop.mp4`, `seg_N_crop_stable.mp4`, `seg_N_blur.mp4`. If the run crashes, re-run the same command; existing temp files are reused and only missing segments are recomputed. Use `--no-resume` to force recomputing all segments (and optionally clear `output/temp/` first).

## Audio

OpenCV writes video only, so the assembled result first lands in `output/temp/{output_stem}_video_only.mp4`; when ffmpeg is on PATH, `audio_mux.py` muxes the original audio in (full track in `--full-timeline` mode, per-segment trims in concat mode) and re-encodes to H.264/AAC at the final output path. If ffmpeg is missing, the input is silent, or the mux fails, the video-only file is moved to the output path instead. `--no-audio` skips the mux entirely.

## Dependency pins (do not loosen)

OpenCV must stay `<5`: OpenCV 5 removed `cv2.BRISK_create` from the main module, which breaks `imutils` (a vidstab dependency) at import time. scenedetect must stay `<0.7`: 0.7 hard-depends on the GUI `opencv-python` wheel, installing a second conflicting `cv2` next to the pinned headless one. Exactly one OpenCV distribution (`opencv-contrib-python-headless`) should be installed; `tests/test_pipeline_e2e.py` has a stabilize smoke test that catches these breakages.

## Config paths

- `INPUT_VIDEO` / `OUTPUT_VIDEO` in `config.py` are defaults; `pipeline.py` accepts `--input` and `--output` to override. If the default input path doesn’t exist, the pipeline exits with an error until the user provides a valid path.

## Debugging

- `debug_blur.py --segment 0 --save --cropped`: preview crop-then-blur on the first pillarboxed segment; writes images to `output/debug_blur/`.
- `debug_blur.py --dump`: writes per-column mean CSV for pillarbox detection analysis.
- Pillarbox bounds use a horizontal window (`PILLAR_EDGE_WINDOW`) and content intersection across sampled frames (max left, min right) to avoid including black bars and to tolerate 1px artifacts.
