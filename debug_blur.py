#!/usr/bin/env python3
"""
Debug pillarbox detection and blur-fill on one or a few frames.
Prints bounds, edge intensities, and optional images so you can see why
blur-fill might not be removing black bars.

Usage:
  python debug_blur.py [input_video] [--frame N] [--segment K] [--save]
  --frame N   : use frame N from the video (default: 0)
  --segment K : use first frame of K-th pillarboxed segment (from cache); overrides --frame
  --save      : write debug images to output/debug_blur/
  --cropped   : use crop-then-blur pipeline (crop to content, then blur_fill_cropped_frame); saved filled image matches pipeline.
  --thresholds 25 40 60 : also report bounds for these black thresholds
  --dump              : write per-column mean array to output/debug_blur/frameN_column_means.csv for analysis
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from config import (
    OUTPUT_DIR,
    BLACK_THRESHOLD,
    PILLAR_MIN_WIDTH_RATIO,
    PILLAR_MIN_CONTENT_RATIO,
    PILLAR_EDGE_WINDOW,
)
from scene_utils import pillarbox_content_bounds, load_scenes_cache


def column_means_array(frame: np.ndarray, window: int = PILLAR_EDGE_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """
    For each column x, return raw mean (single column) and windowed mean (avg over x±window).
    Returns (raw_means, windowed_means) each shape (w,).
    """
    h, w = frame.shape[:2]
    raw = np.zeros(w, dtype=np.float64)
    windowed = np.zeros(w, dtype=np.float64)
    for x in range(w):
        raw[x] = np.mean(frame[:, x, :])
        x0 = max(0, x - window)
        x1 = min(w, x + window + 1)
        windowed[x] = np.mean(frame[:, x0:x1, :])
    return raw, windowed


def edge_stats(frame: np.ndarray, bar_width: int = 100) -> dict:
    """Mean intensity of left edge, right edge, and center strip."""
    h, w = frame.shape[:2]
    left = frame[:, :bar_width, :]
    right = frame[:, -bar_width:, :]
    mid = frame[:, w // 2 - 50 : w // 2 + 50, :]
    return {
        "left_mean": float(np.mean(left)),
        "left_min": int(np.min(left)),
        "left_max": int(np.max(left)),
        "right_mean": float(np.mean(right)),
        "right_min": int(np.min(right)),
        "right_max": int(np.max(right)),
        "center_mean": float(np.mean(mid)),
    }


def main():
    p = argparse.ArgumentParser(description="Debug pillarbox detection and blur-fill.")
    p.add_argument("input", type=Path, nargs="?", default=None, help="Input video (default: config INPUT_VIDEO)")
    p.add_argument("--frame", type=int, default=None, help="Frame index to inspect (default: 0 or first of segment)")
    p.add_argument("--segment", type=int, default=None, help="Use first frame of N-th pillarboxed segment (from cache)")
    p.add_argument("--save", action="store_true", help="Save debug images to output/debug_blur/")
    p.add_argument("--cropped", action="store_true", help="Use crop-then-blur: crop to content, then blur-fill (matches pipeline)")
    p.add_argument("--thresholds", type=int, nargs="*", default=[], help="Try these black thresholds and print bounds")
    p.add_argument("--dump", action="store_true", help="Write per-column mean array to CSV for analysis")
    args = p.parse_args()
    from config import INPUT_VIDEO
    input_path = args.input or INPUT_VIDEO
    if not input_path.exists():
        print(f"Not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve frame index
    frame_idx = args.frame
    if args.segment is not None:
        classified = load_scenes_cache(input_path)
        if not classified:
            print("No scene cache for this video; run pipeline or inspect_scenes first, or use --frame N", file=sys.stderr)
            sys.exit(1)
        pillarboxed_scenes = [(s, e) for s, e, label in classified if label == "pillarboxed"]
        if args.segment >= len(pillarboxed_scenes):
            print(f"Only {len(pillarboxed_scenes)} pillarboxed segments (0..{len(pillarboxed_scenes)-1})", file=sys.stderr)
            sys.exit(1)
        start, _ = pillarboxed_scenes[args.segment]
        frame_idx = start
        print(f"Using first frame of pillarboxed segment {args.segment}: frame {frame_idx}")
    if frame_idx is None:
        frame_idx = 0

    cap = cv2.VideoCapture(str(input_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"Could not read frame {frame_idx}", file=sys.stderr)
        sys.exit(1)

    h, w = frame.shape[:2]
    min_bar = int(w * PILLAR_MIN_WIDTH_RATIO)
    print(f"\nFrame {frame_idx} | size {w}x{h}")
    print(f"Config: BLACK_THRESHOLD={BLACK_THRESHOLD}, PILLAR_MIN_WIDTH_RATIO={PILLAR_MIN_WIDTH_RATIO} -> min_bar={min_bar} px per side")
    print(f"        PILLAR_EDGE_WINDOW={PILLAR_EDGE_WINDOW} (avg over ±N cols to ignore 1px lines), PILLAR_MIN_CONTENT_RATIO={PILLAR_MIN_CONTENT_RATIO}")
    print()

    # Edge stats
    stats = edge_stats(frame)
    print("Edge intensities (0=black, 255=white):")
    print(f"  Left  (first 100 cols): mean={stats['left_mean']:.1f} min={stats['left_min']} max={stats['left_max']}")
    print(f"  Right (last 100 cols):  mean={stats['right_mean']:.1f} min={stats['right_min']} max={stats['right_max']}")
    print(f"  Center:                 mean={stats['center_mean']:.1f}")
    print()

    # Bounds with default threshold
    bounds = pillarbox_content_bounds(frame, black_threshold=BLACK_THRESHOLD)
    print(f"pillarbox_content_bounds(threshold={BLACK_THRESHOLD}): {bounds}")
    if bounds:
        x_left, x_right = bounds
        content_w = x_right - x_left
        print(f"  -> content region x=[{x_left}, {x_right}) width={content_w} ({100*content_w/w:.1f}% of frame)")
    else:
        print("  -> No pillarbox detected (bounds None) -> blur_fill will just resize whole frame, keeping bars.")
    print()

    # Optional: dump full per-column arrays to CSV
    if args.dump:
        out_dir = OUTPUT_DIR / "debug_blur"
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_means, windowed_means = column_means_array(frame)
        csv_path = out_dir / f"frame{frame_idx}_column_means.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "raw_mean", "windowed_mean", "is_black_windowed"])
            for x in range(len(raw_means)):
                w.writerow([x, f"{raw_means[x]:.4f}", f"{windowed_means[x]:.4f}", windowed_means[x] < BLACK_THRESHOLD])
        print(f"Dumped {len(raw_means)} columns to {csv_path}")

    # Optional: try other thresholds
    if args.thresholds:
        print("Bounds for other thresholds:")
        for th in args.thresholds:
            b = pillarbox_content_bounds(frame, black_threshold=th)
            if b:
                print(f"  threshold={th}: {b} (content width {b[1]-b[0]})")
            else:
                print(f"  threshold={th}: None")

    # Optional: blur_fill result and save images
    if args.save:
        out_dir = OUTPUT_DIR / "debug_blur"
        out_dir.mkdir(parents=True, exist_ok=True)
        from blur_fill import blur_fill_frame

        # Image 1: frame with content bounds rectangle (if detected)
        vis = frame.copy()
        if bounds:
            x_left, x_right = bounds
            cv2.rectangle(vis, (x_left, 0), (x_right - 1, h - 1), (0, 255, 0), 2)
            cv2.putText(vis, f"content [{x_left},{x_right})", (x_left, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "no pillarbox detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        path_bbox = out_dir / f"frame{frame_idx}_bounds.png"
        cv2.imwrite(str(path_bbox), vis)
        print(f"Saved: {path_bbox}")

        # Image 2: blur_fill output (crop-then-blur pipeline vs legacy single-step)
        if args.cropped and bounds:
            cropped = frame[:, bounds[0] : bounds[1], :]
            from blur_fill import blur_fill_cropped_frame
            filled = blur_fill_cropped_frame(cropped)
        else:
            filled = blur_fill_frame(frame)
        path_filled = out_dir / f"frame{frame_idx}_filled.png"
        cv2.imwrite(str(path_filled), filled)
        print(f"Saved: {path_filled}")
        print(f"\nDebug images in {out_dir}")


if __name__ == "__main__":
    main()
