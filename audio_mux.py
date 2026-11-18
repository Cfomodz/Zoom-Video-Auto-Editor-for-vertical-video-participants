"""
Mux audio from the original recording into the processed (silent) video.

OpenCV's VideoWriter cannot carry audio, so the pipeline writes video-only
files. When ffmpeg is on PATH, the final step muxes the original audio back in
and re-encodes the video to H.264 for broad player compatibility.

The processed video is a sequence of source frame ranges: one range for
full-timeline output, one per processed segment for concat output. For each
range the matching audio is cut and the pieces are concatenated, so the same
code path keeps both modes in sync. Audio times are taken relative to the
first video frame (see media.video_start_offset): recorders often start the
audio track earlier than the video, and frame 0 is not audio time 0 then.
"""
import shutil
import subprocess
from pathlib import Path

from media import video_start_offset

FFMPEG_VIDEO_ARGS = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
]
FFMPEG_AUDIO_ARGS = ["-c:a", "aac", "-b:a", "160k"]


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


def _video_codec(path: Path) -> str | None:
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
         "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def audio_filter(segments: list[tuple[int, int]], fps: float, video_offset: float) -> str:
    """
    filter_complex that cuts input 1's audio to the given frame ranges and
    concatenates them into [aout].

    - aresample=async=1:first_pts=0 pads leading silence if the audio starts
      after the video, and fills gaps inside the track.
    - apad + exact atrim per range: every piece is exactly as long as its
      video range even if the audio track ends early, so later pieces can't
      slide out of sync.
    """
    end = max(e for _, e in segments) / fps + video_offset
    n = len(segments)
    parts = [
        f"[1:a]aresample=async=1:first_pts=0,apad=whole_dur={end + 1:.6f},"
        f"asplit={n}" + "".join(f"[s{i}]" for i in range(n))
    ]
    for i, (start_frame, end_frame) in enumerate(segments):
        t0 = video_offset + start_frame / fps
        t1 = video_offset + end_frame / fps
        parts.append(f"[s{i}]atrim=start={t0:.6f}:end={t1:.6f},asetpts=PTS-STARTPTS[a{i}]")
    parts.append("".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]")
    return ";".join(parts)


def mux_segments_audio(
    video_path: Path,
    input_path: Path,
    segments: list[tuple[int, int]],
    fps: float,
    output_path: Path,
):
    """Mux the audio for `segments` (source frame ranges, in output order) under video_path."""
    filter_complex = audio_filter(segments, fps, video_start_offset(input_path))
    # The timeline is normally already H.264 (video_io.VideoWriter); copy it rather
    # than spend another lossy generation. Re-encode only an mp4v fallback file.
    video_args = ["-c:v", "copy"] if _video_codec(video_path) == "h264" else FFMPEG_VIDEO_ARGS
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "0:v:0", "-map", "[aout]",
        *video_args, *FFMPEG_AUDIO_ARGS,
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def mux_audio(
    video_path: Path,
    input_path: Path,
    output_path: Path,
    segments: list[tuple[int, int]],
    fps: float,
) -> bool:
    """
    Mux original audio into video_path, writing the result to output_path.
    `segments` are the source frame ranges the video contains, in order:
    [(first, last)] for a full timeline, one per processed segment for concat.
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
        mux_segments_audio(video_path, input_path, segments, fps, output_path)
    except subprocess.CalledProcessError as exc:
        print(f"  Audio mux failed ({exc}); output will have no audio.")
        return False
    return True
