# Debug findings & fix plan

> **Status: implemented.** All four steps below are done, plus the audio-mux
> follow-up (see `audio_mux.py`). One additional root cause surfaced during
> implementation: scenedetect 0.7 hard-depends on the GUI `opencv-python`
> wheel, dragging OpenCV 5 back into a clean install next to the pinned 4.x —
> fixed by pinning `scenedetect>=0.6,<0.7`. Verified: 31 tests pass from a
> fresh `pip install -r requirements.txt`, and the CLI runs end-to-end with
> stabilization and audio in both output modes.

## Symptom

Running the pipeline on any video with pillarboxed segments crashes on the first
segment during the stabilize step:

```
File ".../imutils/feature/factories.py", line 61, in <module>
    "BRISK": cv2.BRISK_create,
AttributeError: module 'cv2' has no attribute 'BRISK_create'
```

The only way to get output today is `--skip-stabilize` (that path was verified
working end-to-end: detection, classification, crop, blur-fill, concat, and
`--full-timeline` all produce correct output — all 24 unit tests pass and a
synthetic 4-segment video round-trips with correct frame counts and no black
bars).

## Root causes (two independent bugs)

### 1. Unpinned OpenCV now resolves to 5.0, which breaks `imutils`/`vidstab`

`requirements.txt` pins `opencv-python-headless>=4.8.0`. OpenCV 5.0 is now on
PyPI, and it moved `BRISK` (and other feature detectors) out of the main module
into `cv2.xfeatures2d`. `vidstab` imports `imutils` (unmaintained since 2021),
whose `feature/factories.py` references `cv2.BRISK_create` at **import time**,
so `from vidstab import VidStab` raises `AttributeError` before any
stabilization happens. Because `stabilize.py` only catches `ImportError`, the
`AttributeError` propagates and kills the whole pipeline.

A side effect of the current requirements makes this worse: the `vidstab[cv2]`
extra installs `opencv-contrib-python` (GUI build) *alongside*
`opencv-python-headless`, so a fresh install ends up with **three** conflicting
`cv2` distributions (headless, full, contrib) and which one wins is arbitrary.

### 2. `vidstab` calls `cv2.destroyAllWindows()` — crashes on headless builds

Even with OpenCV pinned to 4.x, stabilization still fails at the end of each
segment:

```
File ".../vidstab/VidStab.py", line 299, in _apply_transforms
    cv2.destroyAllWindows()
cv2.error: OpenCV(4.13.0) ... The function is not implemented. Rebuild the
library with Windows, GTK+ 2.x or Cocoa support...
```

`VidStab._apply_transforms()` calls `cv2.destroyAllWindows()` unconditionally
(even with `show_progress=False`), and headless OpenCV builds — which
`requirements.txt` explicitly asks for — raise on any highgui call. The call
happens *before* the writer is released, so the stabilized temp file can be
left unfinalized.

## Fix plan

### Step 1 — pin and dedupe OpenCV in `requirements.txt`

```
opencv-contrib-python-headless>=4.8,<5
numpy>=1.24.0
scenedetect>=0.6.0
vidstab>=1.7.0
tqdm>=4.65.0
```

- One OpenCV distribution only. `opencv-contrib-python-headless<5` keeps
  `BRISK_create` in the main module (what `imutils` expects) and includes the
  contrib modules vidstab can use.
- Drop the `vidstab[cv2]` extra — it's what pulls in the extra GUI
  `opencv-contrib-python`.
- Drop the `scenedetect[opencv]` extra — it no longer exists in scenedetect
  0.7 (pip warns "does not provide the extra 'opencv'") and OpenCV is already
  a direct dependency.

### Step 2 — make `stabilize.py` headless-safe

Before invoking `stabilizer.stabilize(...)`, replace `cv2.destroyAllWindows`
with a wrapper that swallows the "not implemented" `cv2.error` (vidstab shares
the same `cv2` module object, so patching it in `stabilize.py` is sufficient):

```python
def _headless_safe_destroy_all_windows():
    try:
        _original_destroy_all_windows()
    except cv2.error:
        pass  # headless OpenCV has no highgui; nothing to destroy
```

Catching the exception around `stabilize()` instead is NOT safe: the raise
happens before `writer.release()`, so the output would be truncated.

Also broaden the import guard in `stabilize_video` from `ImportError` to
`(ImportError, AttributeError)` with a message pointing at the OpenCV pin, so
a future OpenCV/imutils drift fails with an actionable error instead of a raw
traceback.

### Step 3 — fix latent segment-skip misalignment in `pipeline.py`

Unrelated to the crash but found while debugging: when
`get_segment_content_bounds()` returns `None`, the loop does `continue` without
appending to `processed_paths`, while `classified` still labels that segment
`pillarboxed`. In `--full-timeline` mode, `write_full_timeline()` then indexes
`processed_paths[pillarboxed_idx]` once per pillarboxed segment — every
segment after the skipped one gets the *wrong* clip, and the last one raises
`IndexError`. Fix: when bounds are `None`, re-label that segment `full_width`
in `classified` (copy original frames through) instead of `continue`-ing.

### Step 4 — regression coverage

- Add a smoke test that imports `vidstab` and runs `stabilize_video` on a tiny
  synthetic clip (catches both import-time and headless-highgui breakage).
- Add an end-to-end pipeline test on a small synthetic video (full-width +
  pillarboxed segments) asserting exit code 0 and output frame count.

## Verification already performed

- OpenCV 4.13 + no-op `destroyAllWindows`: full pipeline (crop → stabilize →
  blur-fill → `--full-timeline`) exits 0 on a synthetic 384-frame video;
  output has 384 frames; each temp stage (`seg_N_crop`, `seg_N_crop_stable`,
  `seg_N_blur`) has the expected 96 frames; pillarboxed frames in the output
  are blur-filled edge-to-edge (no black columns).
- Existing 24-test suite passes on OpenCV 4.13.

## Follow-up: audio mux (implemented)

OpenCV's `VideoWriter` cannot carry audio, so the pipeline originally produced
silent mp4v files. `audio_mux.py` now runs as Step 4 when ffmpeg is on PATH:
the original audio is muxed back in (full track for `--full-timeline`;
per-segment `atrim`+`concat` for concat mode so audio stays in sync) and the
video is re-encoded to H.264/AAC. Falls back to the silent video-only file —
with a printed notice — when ffmpeg is missing, the input has no audio stream,
or the mux fails. Opt out with `--no-audio`.
