#!/usr/bin/env python3
"""Print scene list and pillarboxed vs full_width classification. Uses scene cache when available."""
import argparse
import sys
from pathlib import Path

from config import INPUT_VIDEO


def main():
    p = argparse.ArgumentParser(description="Inspect scene list and pillarboxed vs full_width classification.")
    p.add_argument("input", type=Path, nargs="?", default=INPUT_VIDEO, help="Input video path")
    p.add_argument("--refresh", action="store_true", help="Recompute scenes and update cache (ignore existing cache)")
    args = p.parse_args()
    input_path = args.input
    if not input_path.exists():
        print(f"Not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    from scene_utils import detect_scenes, classify_scenes, load_scenes_cache, save_scenes_cache
    print(f"Input: {input_path}")
    if args.refresh:
        classified = None
    else:
        classified = load_scenes_cache(input_path)
    if classified is None:
        scenes = detect_scenes(input_path)
        print(f"Scenes: {len(scenes)} (computed)\n")
        classified = classify_scenes(input_path, scenes)
        save_scenes_cache(input_path, classified)
    else:
        print(f"Scenes: {len(classified)} (from cache)\n")
    for i, (start, end, label) in enumerate(classified):
        print(f"  Scene {i}: frames {start}-{end} -> {label}")
    pillarboxed_count = sum(1 for _, _, l in classified if l == "pillarboxed")
    print(f"\nPillarboxed segments: {pillarboxed_count} / {len(classified)}")


if __name__ == "__main__":
    main()
