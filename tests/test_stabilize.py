"""
Stabilizer tests on a synthetic handheld clip: a portrait window that shakes
over a textured scene while slowly panning (intentional motion).

- Shake is removed (frame-to-frame jitter drops), the pan is kept.
- Frame count and size are preserved.
- No fill pixels at the edges: vidstab's "replicate" border left smeared
  streaks along the content edge that looked like a bad crop.
- Corrections that would need more than STABILIZE_MAX_ZOOM are scaled back.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import stabilize  # noqa: E402
from stabilize import plan_corrections, smooth_path, stabilize_video  # noqa: E402

W, H, N, FPS = 240, 400, 60, 30.0
PAN = 0.6  # px/frame intentional pan
SHAKE = 4.0  # px std of hand shake


def _scene(seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    # Fine texture so neighbouring columns differ clearly (replicated fill makes them identical).
    canvas = cv2.GaussianBlur(rng.randint(40, 200, (H + 200, W + 260, 3)).astype(np.uint8), (0, 0), 1.2)
    for _ in range(300):
        x, y = rng.randint(0, canvas.shape[1] - 30), rng.randint(0, canvas.shape[0] - 30)
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.rectangle(canvas, (x, y), (x + rng.randint(8, 30), y + rng.randint(8, 30)), color, -1)
    return canvas


@pytest.fixture(scope="module")
def shaky_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("stab")
    rng = np.random.RandomState(3)
    scene = _scene()
    path = d / "shaky.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        x = 100 + i * PAN + rng.randn() * SHAKE
        y = 100 + rng.randn() * SHAKE
        m = np.float32([[1, 0, -x], [0, 1, -y]])
        out.write(cv2.warpAffine(scene, m, (W, H), flags=cv2.INTER_LINEAR))
    out.release()
    stable = d / "stable.mp4"
    stabilize_video(path, stable, show_progress=False)
    return path, stable


def _read(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def _shifts(frames) -> np.ndarray:
    g = [np.float32(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
    return np.array([cv2.phaseCorrelate(a, b)[0] for a, b in zip(g, g[1:])])


def test_preserves_frame_count_and_size(shaky_clip):
    src, stable = shaky_clip
    a, b = _read(src), _read(stable)
    assert len(b) == len(a) == N
    assert b[0].shape == a[0].shape


def test_removes_shake_keeps_pan(shaky_clip):
    src, stable = shaky_clip
    before, after = _shifts(_read(src)), _shifts(_read(stable))
    # Jitter: spread of frame-to-frame motion around its typical value.
    jitter_before = np.std(before[:, 0] - np.median(before[:, 0]))
    jitter_after = np.std(after[:, 0] - np.median(after[:, 0]))
    assert jitter_after < jitter_before / 3, (jitter_before, jitter_after)
    # Intentional pan survives (content keeps drifting the same direction).
    assert np.median(after[5:-5, 0]) == pytest.approx(np.median(before[:, 0]), abs=0.6)


def test_no_fill_at_edges(shaky_clip):
    """Replicated border pixels make adjacent edge columns/rows identical.

    Calibration on this clip: with the same corrections but no zoom and a
    replicate border (what vidstab did), 58/60 frames have an edge reading
    below 0.3 (median 0.00); the source clip's median is ~1.5.
    """
    _, stable = shaky_clip
    for f in _read(stable):
        f = f.astype(np.int16)
        edges = [
            np.abs(f[:, 0] - f[:, 1]).mean(), np.abs(f[:, -1] - f[:, -2]).mean(),
            np.abs(f[0] - f[1]).mean(), np.abs(f[-1] - f[-2]).mean(),
        ]
        assert min(edges) > 0.3, edges


def test_smooth_path_is_centered():
    path = np.column_stack([np.arange(20.0), np.zeros(20), np.zeros(20)])
    sm = smooth_path(path, 5)
    # A straight line (steady pan) is unchanged away from the ends: no lag.
    assert np.allclose(sm[3:-3, 0], path[3:-3, 0])


def test_large_shake_is_clamped_to_max_zoom():
    rng = np.random.RandomState(0)
    deltas = np.column_stack([rng.randn(50) * 80, rng.randn(50) * 80, rng.randn(50) * 0.05])
    corr, zoom = plan_corrections(deltas, 600, 1000, 15, 1.12)
    assert 1.0 < zoom <= 1.12 + 1e-9
    mx, my = stabilize._uncovered_margins(corr, 600, 1000)
    assert mx.max() <= 600 * (1 - 1 / zoom) / 2 + 1e-6
    assert my.max() <= 1000 * (1 - 1 / zoom) / 2 + 1e-6


def test_still_clip_needs_no_zoom():
    corr, zoom = plan_corrections(np.zeros((30, 3)), 600, 1000, 15, 1.12)
    assert zoom == 1.0 and not corr.any()


def test_rotation_does_not_leak_into_shift():
    """Motion is measured at the frame center: a pure rotation about the center
    must read as rotation, not as a shift (rotation about the top-left corner
    would add ~distance x angle of fake translation)."""
    frame = cv2.cvtColor(_scene()[:H, :W], cv2.COLOR_BGR2GRAY)
    rot = cv2.warpAffine(frame, cv2.getRotationMatrix2D((W / 2, H / 2), 1.0, 1.0), (W, H))
    dx, dy, da = stabilize._pair_motion(frame, rot, 1.0)
    assert abs(dx) < 0.5 and abs(dy) < 0.5, (dx, dy)
    assert abs(abs(np.degrees(da)) - 1.0) < 0.1, np.degrees(da)


def test_cut_sized_jump_is_ignored():
    frame = cv2.cvtColor(_scene()[:H, :W], cv2.COLOR_BGR2GRAY)
    jumped = np.roll(frame, 60, axis=1)  # 60 px of 400 = 15%: not hand shake
    assert stabilize._pair_motion(frame, jumped, 1.0) == (0.0, 0.0, 0.0)
