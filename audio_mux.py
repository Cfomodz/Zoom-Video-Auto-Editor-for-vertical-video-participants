"""
Mux audio from the original recording into the processed (silent) video.

OpenCV's VideoWriter cannot carry audio, so the pipeline writes video-only
files. When ffmpeg is on PATH, the final step muxes the original audio back in
and re-encodes the video to H.264 for broad player compatibility:
- Full-timeline output has the same duration as the input, so the original
  audio track is mapped in directly.
- Concat output contains only the pillarboxed segments, so the matching audio
  ranges are trimmed from the original and concatenated to stay in sync.
"""
import shutil
import subprocess
from pathlib import Path

FFMPEG_VIDEO_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
FFMPEG_AUDIO_ARGS = ["-c:a", "aac"]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def input_has_audio(input_path: Path) -> bool:
    """True if the input has at least one audio stream (assume yes if ffprobe is missing)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return True
    result = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(input_path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def mux_full_timeline_audio(video_path: Path, input_path: Path, output_path: Path):
    """Mux the full original audio track into a same-duration processed video."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-i", str(input_path),
        "-map", "0:v:0", "-map", "1:a:0",
        *FFMPEG_VIDEO_ARGS, *FFMPEG_AUDIO_ARGS,
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def mux_segments_audio(
    video_path: Path,
    input_path: Path,
    segments: list[tuple[int, int]],
    fps: float,
    output_path: Path,
):
    """
    Mux audio for a concat-mode output: trim the original audio to each
    (start_frame, end_frame) segment and concatenate, so audio lines up with
    the concatenated video segments.
    """
    trims = []
    labels = []
    for i, (start_frame, end_frame) in enumerate(segments):
        trims.append(
            f"[1:a]atrim=start={start_frame / fps:.6f}:end={end_frame / fps:.6f},"
            f"asetpts=PTS-STARTPTS[a{i}]"
        )
        labels.append(f"[a{i}]")
    filter_complex = (
        ";".join(trims) + ";" + "".join(labels) + f"concat=n={len(segments)}:v=0:a=1[aout]"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "0:v:0", "-map", "[aout]",
        *FFMPEG_VIDEO_ARGS, *FFMPEG_AUDIO_ARGS,
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def mux_audio(
    video_path: Path,
    input_path: Path,
    output_path: Path,
    full_timeline: bool,
    segments: list[tuple[int, int]],
    fps: float,
) -> bool:
    """
    Mux original audio into video_path, writing the result to output_path.
    Returns True on success, False if muxing isn't possible (no ffmpeg, no
    audio stream, or ffmpeg failed) — in which case output_path is untouched.
    """
    if not ffmpeg_available():
        print("  ffmpeg not found on PATH; output will have no audio.")
        return False
    if not input_has_audio(input_path):
        print("  Input has no audio stream; skipping audio mux.")
        return False
    try:
        if full_timeline:
            mux_full_timeline_audio(video_path, input_path, output_path)
        else:
            mux_segments_audio(video_path, input_path, segments, fps, output_path)
    except subprocess.CalledProcessError as exc:
        print(f"  Audio mux failed ({exc}); output will have no audio.")
        return False
    return True
