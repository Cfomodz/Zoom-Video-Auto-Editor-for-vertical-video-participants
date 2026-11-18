"""
End-to-end pipeline tests on small synthetic videos.

Covers the regressions found in debugging:
1. Stabilize smoke test — catches vidstab/imutils import breakage (OpenCV 5
   removed cv2.BRISK_create) and vidstab's cv2.destroyAllWindows() call that
   raises on headless OpenCV builds.
2. Full pipeline runs — exit code 0 and correct output frame counts in both
   concat and --full-timeline modes.
3. Segment with no stable pillarbox bounds is relabeled full_width instead of
   desyncing/overflowing processed_paths in full-timeline mode.
4. Audio mux — output carries an audio stream when the input has one.
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

import pipeline  # noqa: E402
import scene_utils  # noqa: E402
from stabilize import stabilize_video  # noqa: E402

W, H, FPS = 640, 360, 24.0
SEG_FRAMES = 60
BAR = 160  # black bar width on each side of pillarboxed segments


def _fullwidth_frame(t: int, color: tuple) -> np.ndarray:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = color
    x = int((t * 4) % (W - 60))
    cv2.rectangle(frame, (x, 100), (x + 60, 200), (0, 200, 255), -1)
    return frame


def _pillarboxed_frame(t: int, content: np.ndarray, left: int = BAR) -> np.ndarray:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, left:left + content.shape[1], :] = content
    x = int((t * 3) % (content.shape[1] - 40))
    cv2.circle(frame, (left + x + 20, 150), 15, (255, 50, 50), -1)
    return frame


def _make_content(seed: int, width: int = W - 2 * BAR) -> np.ndarray:
    rng = np.random.RandomState(seed)
    content = rng.randint(60, 220, (H, width, 3), dtype=np.uint8)
    return cv2.GaussianBlur(content, (31, 31), 0)


def _write_video(path: Path, frames) -> None:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        out.write(f)
    out.release()


def _make_input(path: Path) -> int:
    """Full-width, pillarboxed, full-width, pillarboxed. Returns total frames."""
    frames = []
    content_a = _make_content(1)
    content_b = _make_content(2)
    for t in range(SEG_FRAMES):
        frames.append(_fullwidth_frame(t, (40, 90, 60)))
    for t in range(SEG_FRAMES):
        frames.append(_pillarboxed_frame(t, content_a))
    for t in range(SEG_FRAMES):
        frames.append(_fullwidth_frame(t, (120, 40, 150)))
    for t in range(SEG_FRAMES):
        frames.append(_pillarboxed_frame(t, content_b))
    _write_video(path, frames)
    return len(frames)


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


@pytest.fixture
def isolated_output(tmp_path, monkeypatch):
    """Point pipeline temp files and the scene cache at tmp_path."""
    out_dir = tmp_path / "output"
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(scene_utils, "OUTPUT_DIR", out_dir)
    return out_dir


class TestStabilizeSmoke:
    def test_vidstab_imports(self):
        """imutils references cv2.BRISK_create at import; OpenCV must provide it."""
        from vidstab import VidStab  # noqa: F401

    def test_stabilize_headless_safe(self, tmp_path):
        """stabilize_video must survive vidstab's cv2.destroyAllWindows() call
        on headless OpenCV builds, and write all frames."""
        clip = tmp_path / "shaky.mp4"
        rng = np.random.RandomState(3)
        base = _make_content(4, width=200)
        out = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (200, H))
        for _ in range(40):
            dx, dy = int(rng.randn() * 3), int(rng.randn() * 3)
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            out.write(cv2.warpAffine(base, m, (200, H), borderMode=cv2.BORDER_REPLICATE))
        out.release()
        stable = tmp_path / "stable.mp4"
        stabilize_video(clip, stable, show_progress=False)
        assert stable.exists()
        assert _frame_count(stable) == 40


class TestPipelineEndToEnd:
    def test_concat_mode(self, tmp_path, isolated_output):
        input_path = tmp_path / "input.mp4"
        _make_input(input_path)
        output_path = tmp_path / "out.mp4"
        rc = pipeline.run_pipeline(
            input_path, output_path, skip_stabilize=True, resume=False, with_audio=False
        )
        assert rc == 0
        assert output_path.exists()
        # Only the two pillarboxed segments, blur-filled to full output size
        assert _frame_count(output_path) == 2 * SEG_FRAMES

    def test_full_timeline_mode(self, tmp_path, isolated_output):
        input_path = tmp_path / "input.mp4"
        total = _make_input(input_path)
        output_path = tmp_path / "out.mp4"
        rc = pipeline.run_pipeline(
            input_path,
            output_path,
            skip_stabilize=True,
            full_timeline=True,
            resume=False,
            with_audio=False,
        )
        assert rc == 0
        assert _frame_count(output_path) == total

    def test_unstable_bounds_segment_relabeled(self, tmp_path, isolated_output):
        """A scene whose frames are individually pillarboxed but whose content
        regions don't intersect (bounds=None) must be treated as full_width,
        not desync write_full_timeline's processed_paths indexing."""
        input_path = tmp_path / "input.mp4"
        frames = []
        content = _make_content(5, width=200)
        for t in range(SEG_FRAMES):
            frames.append(_fullwidth_frame(t, (40, 90, 60)))
        # "Jumping" pillarboxed scene: content alternates between far left and
        # far right, so the per-frame bounds never intersect across samples.
        for t in range(SEG_FRAMES):
            left = 60 if t % 2 == 0 else W - 60 - 200
            frames.append(_pillarboxed_frame(t, content, left=left))
        # A normal pillarboxed scene afterwards — its processed clip must land
        # in the right place in the timeline.
        content_b = _make_content(6)
        for t in range(SEG_FRAMES):
            frames.append(_pillarboxed_frame(t, content_b))
        _write_video(input_path, frames)

        output_path = tmp_path / "out.mp4"
        rc = pipeline.run_pipeline(
            input_path,
            output_path,
            skip_stabilize=True,
            full_timeline=True,
            resume=False,
            with_audio=False,
        )
        assert rc == 0
        assert _frame_count(output_path) == len(frames)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
class TestAudioMux:
    def _add_audio(self, video_path: Path, out_path: Path, duration: float):
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(video_path),
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(out_path),
            ],
            check=True,
        )

    def _audio_duration(self, path: Path) -> float:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        assert out.stdout.strip(), f"No audio stream in {path}"
        return float(out.stdout.strip())

    def test_full_timeline_keeps_audio(self, tmp_path, isolated_output):
        silent = tmp_path / "silent.mp4"
        total = _make_input(silent)
        input_path = tmp_path / "input.mp4"
        self._add_audio(silent, input_path, duration=total / FPS)
        output_path = tmp_path / "out.mp4"
        rc = pipeline.run_pipeline(
            input_path, output_path, skip_stabilize=True, full_timeline=True, resume=False
        )
        assert rc == 0
        assert abs(self._audio_duration(output_path) - total / FPS) < 0.5

    def test_concat_mode_audio_matches_segments(self, tmp_path, isolated_output):
        silent = tmp_path / "silent.mp4"
        total = _make_input(silent)
        input_path = tmp_path / "input.mp4"
        self._add_audio(silent, input_path, duration=total / FPS)
        output_path = tmp_path / "out.mp4"
        rc = pipeline.run_pipeline(input_path, output_path, skip_stabilize=True, resume=False)
        assert rc == 0
        # Audio should cover exactly the two pillarboxed segments
        expected = 2 * SEG_FRAMES / FPS
        assert abs(self._audio_duration(output_path) - expected) < 0.5
