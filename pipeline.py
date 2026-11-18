#!/usr/bin/env python3
"""
Pipeline: detect scenes, classify pillarboxed vs full-width segments, process
pillarboxed segments (crop → stabilize → blur-fill), output full timeline or
pillarboxed-only concat. Temp files are written per segment for resumability.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2

from config import INPUT_VIDEO, OUTPUT_DIR, OUTPUT_VIDEO  # INPUT_VIDEO can be overridden by --input
from scene_utils import (
    detect_scenes,
    classify_scenes,
    load_scenes_cache,
    save_scenes_cache,
)
from crop import get_segment_content_bounds, crop_segment
from blur_fill import blur_fill_segment_from_cropped
from stabilize import stabilize_video
from audio_mux import mux_audio


def extract_fps_and_frame_count(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, count


def concat_videos(segment_paths: list[Path], output_path: Path, fps: float):
    """Concatenate segment video files into one output (same resolution/codec). Streams frame-by-frame; no full load in memory."""
    if not segment_paths:
        return
    cap0 = cv2.VideoCapture(str(segment_paths[0]))
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for seg_path in segment_paths:
        cap = cv2.VideoCapture(str(seg_path))
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            out.write(frame)
        cap.release()
    out.release()


def write_full_timeline(
    input_path: Path,
    classified: list[tuple[int, int, str]],
    processed_paths: list[Path],
    output_path: Path,
    fps: float,
    w: int,
    h: int,
):
    """
    Full timeline: full_width segments from original, pillarboxed segments from processed temp files.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    cap_input = cv2.VideoCapture(str(input_path))
    pillarboxed_idx = 0
    for start_frame, end_frame, label in classified:
        n_frames = end_frame - start_frame
        if label == "full_width":
            cap_input.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(n_frames):
                ret, frame = cap_input.read()
                if not ret or frame is None:
                    break
                # Ensure size in case of any mismatch
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                out.write(frame)
        else:
            seg_path = processed_paths[pillarboxed_idx]
            pillarboxed_idx += 1
            cap_seg = cv2.VideoCapture(str(seg_path))
            for _ in range(n_frames):
                ret, frame = cap_seg.read()
                if not ret or frame is None:
                    break
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                out.write(frame)
            cap_seg.release()
    cap_input.release()
    out.release()


def run_pipeline(
    input_path: Path,
    output_path: Path,
    skip_stabilize: bool = False,
    full_timeline: bool = False,
    resume: bool = True,
    with_audio: bool = True,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    classified = load_scenes_cache(input_path)
    if classified is None:
        print("Step 1: Detecting scenes…")
        scene_list = detect_scenes(input_path)
        print(f"  Found {len(scene_list)} scenes")
        print("Step 2: Classifying pillarboxed vs full_width segments…")
        classified = classify_scenes(input_path, scene_list)
        save_scenes_cache(input_path, classified)
    else:
        print("Step 1 & 2: Using cached scene list (same video).")
    pillarboxed_segments = [
        (i, s, e) for i, (s, e, label) in enumerate(classified) if label == "pillarboxed"
    ]
    print(f"  Pillarboxed segments: {len(pillarboxed_segments)}")

    if not pillarboxed_segments:
        print("No pillarboxed segments found. Check PILLAR_* and BLACK_THRESHOLD in config.")
        return 1

    fps, total_frames = extract_fps_and_frame_count(input_path)
    cap0 = cv2.VideoCapture(str(input_path))
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    processed_paths = []
    processed_segments = []  # (start_frame, end_frame) matching processed_paths, for audio mux
    for idx, (ci, start_frame, end_frame) in enumerate(pillarboxed_segments):
        crop_path = temp_dir / f"seg_{idx}_crop.mp4"
        stable_crop_path = temp_dir / f"seg_{idx}_crop_stable.mp4"
        blur_path = temp_dir / f"seg_{idx}_blur.mp4"
        if resume and blur_path.exists():
            print(f"  Reusing segment {idx + 1}/{len(pillarboxed_segments)} (frames {start_frame}-{end_frame})…")
            processed_paths.append(blur_path)
            processed_segments.append((start_frame, end_frame))
            continue
        print(f"  Processing segment {idx + 1}/{len(pillarboxed_segments)} (frames {start_frame}-{end_frame})…")
        # Step A: crop to vertical content only (no black bars)
        if not (resume and crop_path.exists()):
            bounds = get_segment_content_bounds(input_path, start_frame, end_frame)
            if bounds is None:
                # Relabel so write_full_timeline copies the original frames instead
                # of consuming a processed path that was never produced.
                print(f"    Warning: no stable pillarbox bounds for segment {idx}; treating as full_width.")
                classified[ci] = (start_frame, end_frame, "full_width")
                continue
            crop_segment(input_path, start_frame, end_frame, crop_path, bounds, fps)
        # Step B: stabilize the cropped (raw) video so motion reduction sees the real shake
        source_for_blur = crop_path
        if not skip_stabilize:
            if not (resume and stable_crop_path.exists()):
                stabilize_video(crop_path, stable_crop_path, show_progress=False)
            source_for_blur = stable_crop_path
        # Step C: blur-fill into 1920x1080 (from crop or stabilized crop)
        if not (resume and blur_path.exists()):
            blur_fill_segment_from_cropped(source_for_blur, blur_path, fps, progress_callback=None)
        processed_paths.append(blur_path)
        processed_segments.append((start_frame, end_frame))

    if not processed_paths:
        print("No segments could be processed. Check PILLAR_* and BLACK_THRESHOLD in config.")
        return 1

    # OpenCV writes video only; when muxing audio, write to a temp file first.
    video_only_path = temp_dir / f"{output_path.stem}_video_only.mp4" if with_audio else output_path
    if full_timeline:
        print("Step 3: Writing full timeline…")
        write_full_timeline(input_path, classified, processed_paths, video_only_path, fps, w, h)
    else:
        print("Step 3: Concatenating processed segments…")
        concat_videos(processed_paths, video_only_path, fps)

    if with_audio:
        print("Step 4: Muxing original audio (ffmpeg)…")
        if mux_audio(video_only_path, input_path, output_path, full_timeline, processed_segments, fps):
            video_only_path.unlink()
        else:
            shutil.move(str(video_only_path), str(output_path))
    print(f"  Written: {output_path}")
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Process conference recording: crop, stabilize, and blur-fill pillarboxed segments."
    )
    p.add_argument("--input", type=Path, default=INPUT_VIDEO, help="Input video path")
    p.add_argument("--output", type=Path, default=OUTPUT_VIDEO, help="Output video path")
    p.add_argument("--skip-stabilize", action="store_true", help="Only blur-fill, no motion smoothing")
    p.add_argument(
        "--full-timeline",
        action="store_true",
        help="Output full-length video: full_width unchanged, pillarboxed segments processed.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute all segments; do not reuse existing temp files.",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip the ffmpeg audio mux step; output is video-only (mp4v).",
    )
    args = p.parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1
    return run_pipeline(
        args.input,
        args.output,
        skip_stabilize=args.skip_stabilize,
        full_timeline=args.full_timeline,
        resume=not args.no_resume,
        with_audio=not args.no_audio,
    )


if __name__ == "__main__":
    sys.exit(main())
