"""
Audio/video sync regression tests ("clapper" clips).

A white square flashes in the middle of the frame and a 1 kHz beep plays at the
same instants (t = k + 0.5 s), across alternating full-width and pillarboxed
segments. After the pipeline runs, every beep in the output must line up with
a flash within one frame — in both output modes and for the input timing
quirks real recordings have:

- audio stream starting before the first video frame (stream start offset);
- variable frame rate (Zoom drops frames while the picture is static).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import media  # noqa: E402
import pipeline  # noqa: E402
import scene_utils  # noqa: E402
from audio_mux import audio_filter  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

W, H, FPS, SEG = 640, 360, 30, 2  # four 2 s segments
DUR = 4 * SEG
FRAME_MS = 1000 / FPS


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def _make_clapper(path: Path) -> None:
    dim = "colorchannelmixer=rr=.6:gg=.6:bb=.6"  # keep the pattern well below flash white
    full = f"testsrc2=s={W}x{H}:r={FPS}:d={SEG},{dim}"
    pil = f"testsrc2=s=202x{H}:r={FPS}:d={SEG},{dim},pad={W}:{H}:(ow-iw)/2:0:black"
    flash = (
        f"drawbox=x={W}/2-40:y={H}/2-40:w=80:h=80:color=white:t=fill:"
        f"enable='lt(abs(mod(t\\,1)-0.5)\\,0.5/{FPS})'"
    )
    graph = (
        f"{full}[a0];{pil}[a1];{full},hue=h=90[a2];{pil},hue=h=200[a3];"
        f"[a0][a1][a2][a3]concat=n=4:v=1:a=0,{flash},format=yuv420p[v];"
        f"aevalsrc='if(between(mod(t\\,1)\\,0.5\\,0.53)\\,0.8*sin(2*PI*1000*t)\\,0)':s=48000:d={DUR}[a]"
    )
    _ffmpeg("-filter_complex", graph, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", str(path))


def _make_audio_lead(base: Path, path: Path) -> None:
    """Audio stream starts 0.5 s before the first video frame; still in sync on playback."""
    _ffmpeg("-i", str(base), "-itsoffset", "0.5", "-i", str(base),
            "-filter_complex", "[0:a]adelay=500:all=1[a]", "-map", "1:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", str(path))


def _make_vfr(base: Path, path: Path) -> None:
    """Drop every other frame between 3 s and 5 s (keeping flashes), timestamps preserved."""
    keep = "not(between(t\\,3\\,5))+eq(mod(n\\,2)\\,0)+lt(abs(mod(t\\,1)-0.5)\\,0.5/30)"
    _ffmpeg("-i", str(base), "-vf", f"select='{keep}'", "-fps_mode", "vfr",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "copy", str(path))


def _flash_times(path: Path) -> list[float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    times, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if frame[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10].min() > 200:
            times.append(i / fps)
        i += 1
    cap.release()
    return times


def _beep_times(path: Path) -> list[float]:
    """Beep onsets on the playback timeline (audio start offset vs video applied)."""
    a_off = -media.video_start_offset(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", "8000",
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    hot = np.where(np.abs(np.frombuffer(raw, dtype=np.float32)) > 0.3)[0]
    times, last = [], -10**9
    for s in hot:
        if s - last > 800:
            times.append(s / 8000 + a_off)
        last = s
    return times


def _offsets_ms(path: Path) -> list[float]:
    flashes, beeps = _flash_times(path), _beep_times(path)
    assert flashes and beeps, f"no flashes ({len(flashes)}) or beeps ({len(beeps)}) in {path.name}"
    assert len(flashes) == len(beeps), f"{len(flashes)} flashes vs {len(beeps)} beeps"
    return [1000 * (min(flashes, key=lambda f: abs(f - b)) - b) for b in beeps]


@pytest.fixture(scope="module")
def clapper(tmp_path_factory):
    d = tmp_path_factory.mktemp("clapper")
    base = d / "base.mp4"
    _make_clapper(base)
    _make_audio_lead(base, d / "lead.mp4")
    _make_vfr(base, d / "vfr.mp4")
    return d


@pytest.fixture
def isolated_output(tmp_path, monkeypatch):
    out_dir = tmp_path / "output"
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(scene_utils, "OUTPUT_DIR", out_dir)
    return out_dir


class TestInputTiming:
    def test_inputs_are_in_sync_on_playback(self, clapper):
        for name in ("base", "lead"):
            assert all(abs(o) <= FRAME_MS for o in _offsets_ms(clapper / f"{name}.mp4")), name

    def test_video_start_offset(self, clapper):
        assert media.video_start_offset(clapper / "base.mp4") == pytest.approx(0.0, abs=1e-3)
        assert media.video_start_offset(clapper / "lead.mp4") == pytest.approx(0.5, abs=1e-3)

    def test_vfr_detection(self, clapper):
        assert not media.is_variable_frame_rate(clapper / "base.mp4")
        assert not media.is_variable_frame_rate(clapper / "lead.mp4")
        assert media.is_variable_frame_rate(clapper / "vfr.mp4")

    def test_ensure_cfr_passthrough_and_convert(self, clapper, tmp_path):
        assert media.ensure_cfr(clapper / "base.mp4", tmp_path, "k") == clapper / "base.mp4"
        cfr = media.ensure_cfr(clapper / "vfr.mp4", tmp_path, "k")
        assert cfr != clapper / "vfr.mp4" and cfr.exists()
        assert not media.is_variable_frame_rate(cfr)


class TestAudioFilter:
    def test_ranges_are_offset_by_video_start(self):
        graph = audio_filter([(0, 30), (60, 90)], 30.0, 0.5)
        assert "atrim=start=0.500000:end=1.500000" in graph
        assert "atrim=start=2.500000:end=3.500000" in graph
        assert "concat=n=2:v=0:a=1[aout]" in graph


@pytest.mark.parametrize("name", ["base", "lead", "vfr"])
@pytest.mark.parametrize("full_timeline", [True, False], ids=["full_timeline", "concat"])
def test_output_av_sync(clapper, tmp_path, isolated_output, name, full_timeline):
    out = tmp_path / "out.mp4"
    rc = pipeline.run_pipeline(
        clapper / f"{name}.mp4", out, skip_stabilize=True, full_timeline=full_timeline, resume=False
    )
    assert rc == 0
    offsets = _offsets_ms(out)
    # Full timeline keeps every flash; concat keeps the two pillarboxed segments.
    assert len(offsets) == (DUR if full_timeline else DUR // 2)
    assert all(abs(o) <= FRAME_MS for o in offsets), offsets
