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

The assembled video first lands in `output/temp/{output_stem}_video_only.mp4`; when ffmpeg is on PATH, `audio_mux.py` cuts the original audio to the same source frame ranges (one range for `--full-timeline`, one per processed segment in concat mode), concatenates them, and muxes them in (video stream-copied when it is already H.264). If ffmpeg is missing, the input is silent, or the mux fails, the video-only file is moved to the output path instead. `--no-audio` skips the mux entirely.

## Timing and sync

Everything addresses video by frame index (time = index / fps), so:

- `media.ensure_cfr` runs first: variable-frame-rate input (common for Zoom) is converted once to `output/temp/{video_hash}_cfr.mkv` (audio copied) and the rest of the run uses that copy.
- `media.video_start_offset`: when the audio stream starts before the first video frame, audio cuts are shifted by that offset.
- `pipeline.write_exact_frames`: each range contributes exactly `end - start` frames (last frame repeated if a decoder comes up short), so the video timeline matches the audio cut for it.
- `tests/test_av_sync.py` checks flash/beep alignment within one frame for CFR, audio-leads-video, and VFR inputs in both modes.

## Intermediate encoding

`video_io.VideoWriter` pipes frames to ffmpeg/libx264 (near-lossless 4:4:4 for intermediates, 4:2:0 for the timeline/output) and falls back to OpenCV mp4v without ffmpeg. Repeated mp4v generations lost ~10 dB PSNR on real footage. vidstab still writes its stabilized temp file with mp4v (one generation).

## Real-footage sample

`tests/make_real_sample.py` builds `output/real_sample/real_sample.mp4` from Wikimedia Commons clips (real Meet webcam segments + real vertical phone video, with audio), fetching only the needed seconds via ffmpeg range requests. Commons rate-limits bursts (HTTP 429 with Retry-After); the script waits it out. Never commit the result.

## Dependency pins (do not loosen)

`requirements.txt` pins exact versions, tested on Python 3.10–3.13 (numpy has a per-Python-version pin because numpy 2.5 needs 3.12+). OpenCV must stay `<5` (re-checked against 5.1.0): OpenCV 5 removed `cv2.BRISK_create` from the main module, which breaks `imutils` (a vidstab dependency) at import time. scenedetect must stay `<0.7`: 0.7 hard-depends on the GUI `opencv-python` wheel, installing a second conflicting `cv2` next to the pinned headless one. Exactly one OpenCV distribution (`opencv-contrib-python-headless`) should be installed; `tests/test_pipeline_e2e.py` has a stabilize smoke test that catches these breakages.

## Config paths

- `INPUT_VIDEO` / `OUTPUT_VIDEO` in `config.py` are defaults; `pipeline.py` accepts `--input` and `--output` to override. If the default input path doesn’t exist, the pipeline exits with an error until the user provides a valid path.

## Debugging

- `debug_blur.py --segment 0 --save --cropped`: preview crop-then-blur on the first pillarboxed segment; writes images to `output/debug_blur/`.
- `debug_blur.py --dump`: writes per-column mean CSV for pillarbox detection analysis.
- Pillarbox bounds use a horizontal window (`PILLAR_EDGE_WINDOW`) and content intersection across sampled frames (max left, min right) to avoid including black bars and to tolerate 1px artifacts.
