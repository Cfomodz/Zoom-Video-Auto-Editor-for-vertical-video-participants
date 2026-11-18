#!/usr/bin/env python3
"""
Build a real-footage test recording for the pipeline from freely licensed
Wikimedia Commons videos.

The result mimics a speaker-view call recording (1920x1080, H.264 + AAC):
real Google Meet webcam segments (full width) alternate with real vertical
phone footage shown the way call apps display it (centered at full height
on black pillarbox bars). One phone segment carries a Meet/Zoom-style name
label in the bottom-left corner, which lands inside the black bar.

    python tests/make_real_sample.py            # -> output/real_sample/real_sample.mp4

Only the needed seconds of each source are fetched (ffmpeg HTTP range requests,
sequential, retried on HTTP 429) and cached under output/real_sample/src/. Nothing here
is committed: output/ is gitignored. Credits are written next to the video.
"""
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "real_sample"
SRC_DIR = OUT_DIR / "src"
OUTPUT = OUT_DIR / "real_sample.mp4"
UA = "video-pipeline-test-fetch/1.0 (python-urllib)"
FPS = 30

# File titles on Wikimedia Commons.
MEET_CALL = "Wikimania scholarships orientation call recording.webm"  # CC0, Google Meet recording
TUVAN = "WIKITONGUES- Aydyn speaking Tuvan.webm"  # vertical phone video, pillarboxed by its uploader
SELFIE = "President Biden records a selfie video with Shawn Fain of the UAW.webm"  # handheld vertical
MING = "Ming speaking Tày.webm"  # vertical phone video

# Only these time ranges are fetched (ffmpeg HTTP range requests), never whole files:
# name -> (Commons title, start_s, duration_s)
CHUNKS = {
    "meet_a": (MEET_CALL, 1560.0, 40.0),
    "meet_b": (MEET_CALL, 2228.0, 20.0),
    "tuvan": (TUVAN, 55.0, 20.0),
    "selfie": (SELFIE, 0.0, 34.0),
    "ming": (MING, 0.0, 12.0),
}

# (chunk, start_s within chunk, duration_s, layout, name_label)
#   layout "full":     source is already 16:9 (scaled to 1920x1080)
#   layout "portrait": vertical source, scaled to 1080 high, centered on black
SEGMENTS = [
    ("meet_a", 6.0, 12.0, "full", None),
    ("tuvan", 5.0, 14.0, "full", None),  # already pillarboxed in the source
    ("meet_b", 3.0, 12.0, "full", None),
    ("selfie", 4.0, 20.0, "portrait", "Phone participant"),
    ("ming", 0.5, 10.0, "portrait", None),
    ("meet_a", 24.0, 10.0, "full", None),
]


def _request(url: str, *, headers=None, attempts: int = 8):
    """urlopen with Retry-After handling for 429/503."""
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == attempts - 1:
                raise
            wait = int(exc.headers.get("Retry-After") or 0) or 10 * (attempt + 1)
            print(f"  HTTP {exc.code}; waiting {wait}s (Retry-After)…", flush=True)
            time.sleep(min(wait, 900))
    raise RuntimeError("unreachable")


def commons_info(title: str) -> dict:
    params = dict(
        action="query",
        format="json",
        titles=f"File:{title}",
        prop="imageinfo",
        iiprop="url|size|extmetadata",
        iiextmetadatafilter="Artist|LicenseShortName|Credit",
    )
    with _request("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)) as r:
        data = json.load(r)
    page = next(iter(data["query"]["pages"].values()))
    ii = page["imageinfo"][0]
    meta = ii.get("extmetadata", {})

    def text(key):
        raw = meta.get(key, {}).get("value", "")
        return re.sub(r"<[^>]+>", "", raw).strip()

    return {
        "title": title,
        "url": ii["url"],
        "page": ii.get("descriptionurl", ""),
        "license": text("LicenseShortName"),
        "artist": text("Artist"),
        "width": ii.get("width"),
        "height": ii.get("height"),
        "duration": ii.get("duration"),
    }


def fetch_chunk(url: str, start: float, duration: float, dest: Path) -> None:
    """Fetch only [start, start+duration) of a long remote video (ffmpeg HTTP range seeks)."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"  fetching {dest.name} ({start:.0f}s +{duration:.0f}s)…", flush=True)
    tmp = dest.with_name(dest.stem + ".part" + dest.suffix)
    for attempt in range(6):
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-user_agent", UA,
                "-ss", f"{start}", "-i", url, "-t", f"{duration}",
                "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", str(tmp),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            tmp.rename(dest)
            return
        wait = 120 * (attempt + 1) if "429" in result.stderr else 15
        print(f"  ffmpeg fetch failed ({result.stderr.strip()[:120]}); retrying in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"could not fetch {dest.name}")


def has_audio(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def assemble(paths: dict[str, Path]) -> None:
    inputs, chains, concat_in = [], [], []
    for i, (src, start, dur, layout, label) in enumerate(SEGMENTS):
        path = paths[src]
        inputs += ["-ss", f"{start}", "-t", f"{dur}", "-i", str(path)]
        if layout == "full":
            v = f"[{i}:v]scale=1920:1080,setsar=1,fps={FPS},format=yuv420p"
        else:
            v = (
                f"[{i}:v]scale=-2:1080,pad=1920:1080:(ow-iw)/2:0:black,setsar=1,"
                f"fps={FPS},format=yuv420p"
            )
        if label:
            v += (
                f",drawtext=text='{label}':x=24:y=h-56:fontsize=28:fontcolor=white:"
                "box=1:boxcolor=black@0.45:boxborderw=8"
            )
        chains.append(v + f"[v{i}]")
        if has_audio(path):
            chains.append(
                f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,"
                f"apad,atrim=0:{dur},asetpts=PTS-STARTPTS[a{i}]"
            )
        else:
            chains.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{dur}[a{i}]")
        concat_in.append(f"[v{i}][a{i}]")
    graph = ";".join(chains) + ";" + "".join(concat_in) + f"concat=n={len(SEGMENTS)}:v=1:a=1[v][a]"
    tmp = OUTPUT.with_name(OUTPUT.stem + ".part.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", *inputs,
            "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp),
        ],
        check=True,
    )
    tmp.rename(OUTPUT)


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is required", file=sys.stderr)
        return 1
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    infos, paths = {}, {}
    for name, (title, start, dur) in CHUNKS.items():
        if title not in infos:
            infos[title] = commons_info(title)
        dest = SRC_DIR / f"{name}.webm"
        fetch_chunk(infos[title]["url"], start, dur, dest)
        paths[name] = dest
        time.sleep(2)  # be polite between requests
    credits = list(infos.values())
    print("  assembling…", flush=True)
    assemble(paths)
    lines = ["Real-footage test recording assembled from Wikimedia Commons:", ""]
    for c in credits:
        lines.append(f"- {c['title']} — {c['artist'] or 'unknown author'} — {c['license']}")
        lines.append(f"  {c['page']}")
    lines += ["", "Segments (source, start s, duration s, layout, label):"]
    lines += [f"- {s}" for s in SEGMENTS]
    (OUT_DIR / "CREDITS.txt").write_text("\n".join(lines) + "\n")
    print(f"Written: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
