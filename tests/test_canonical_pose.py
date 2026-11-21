import numpy as np
import pytest

from backend.services.canonical_pose import (
    NUM_POINTS,
    adapt_model_output,
    adapt_repo_pose,
    decode_bin,
    encode_bin,
    fill_missing,
    invert_repo_normalization,
    load_layout,
    source_to_canonical,
)


def _canonical_fixture(T=12):
    rng = np.random.default_rng(3)
    pts = rng.normal(0, 0.25, (T, NUM_POINTS, 2)).astype(np.float32)
    # Plausible torso anchors so shoulder width is well defined.
    pts[:, 1] = [0.0, 0.0]
    pts[:, 2] = [-0.5, -0.06]
    pts[:, 5] = [0.5, -0.06]
    pts[:, 8] = [0.0, -1.25]
    return pts


def _to_repo_pixels(points, width=1280.0, height=720.0, shoulder_px=400.0):
    """Canonical -> repository [T,274] values (x/width, y/height, y down)."""
    layout = load_layout("signlingo_openpose137")
    inv = layout["canonicalToSource"]
    shifted = points * shoulder_px
    x = shifted[..., 0] / width
    y = 1.0 - shifted[..., 1] / height
    repo = np.stack([x, y], axis=-1)[:, inv, :]
    return repo.reshape(points.shape[0], 274)


def test_invert_normalization_restores_pixels():
    points = _canonical_fixture()
    repo = _to_repo_pixels(points)
    canon = source_to_canonical(repo.reshape(12, 137, 2), "signlingo_openpose137")
    pixels, missing = invert_repo_normalization(canon)
    assert not missing.any()
    # Shoulder distance in pixel space is the 400 px we encoded.
    width = np.linalg.norm(pixels[0, 5] - pixels[0, 2])
    assert width == pytest.approx(400.0, rel=1e-5)


def test_adapt_repo_pose_roundtrip():
    points = _canonical_fixture()
    repo = _to_repo_pixels(points)
    seq = adapt_repo_pose(repo, source_layout="signlingo_openpose137", source="real", fps=30)
    assert seq.points.shape == (12, 137, 2)
    assert seq.fps == 30
    assert seq.normalization == "neck_origin_shoulder_width"
    # Recovery is exact up to the median shoulder width estimate.
    assert np.allclose(seq.points, points, atol=2e-3)


def test_norm_stats_inversion():
    points = _canonical_fixture()
    repo = _to_repo_pixels(points)
    mean = np.full(274, 0.1, dtype=np.float32)
    std = np.full(274, 0.2, dtype=np.float32)
    normalized = (repo - mean) / std
    seq = adapt_repo_pose(
        normalized,
        source_layout="signlingo_openpose137",
        source="real",
        norm_mean=mean,
        norm_std=std,
    )
    assert np.allclose(seq.points, points, atol=2e-3)


def test_missing_points_are_interpolated():
    points = _canonical_fixture()
    repo = _to_repo_pixels(points).reshape(12, 137, 2)
    # Drop the left wrist for two frames (short gap).
    repo[4, 7] = 0.0
    repo[5, 7] = 0.0
    seq = adapt_repo_pose(repo.reshape(12, 274), source_layout="signlingo_openpose137", source="real")
    # The repaired wrist should sit near the linear interpolation of its neighbours.
    expected = 0.5 * (points[3, 7] + points[6, 7])
    assert np.linalg.norm(seq.points[4, 7] - expected) < 0.15
    assert seq.meta["repairedPoints"] == 2


def test_long_gap_holds_last_valid():
    pts = np.zeros((10, 3, 2), dtype=np.float32)
    valid = np.ones((10, 3), dtype=bool)
    pts[:, :, 0] = np.arange(10)[:, None]
    valid[2:8, 1] = False
    out, repaired = fill_missing(pts, valid, max_gap=2)
    assert repaired[2:8, 1].all()
    assert out[2, 1, 0] == pytest.approx(1.0)
    assert out[7, 1, 0] == pytest.approx(1.0)


def test_generated_output_treats_all_points_valid():
    points = _canonical_fixture()
    repo = _to_repo_pixels(points).reshape(12, 137, 2)
    repo[3, 7] = 0.0
    generated = adapt_model_output(repo.reshape(12, 274), source_layout="openpose137")
    assert generated.source == "generated"
    assert np.all(generated.valid)
    # Nothing was repaired and the result is still in canonical units.
    assert generated.meta["repairedPoints"] == 0
    width = np.linalg.norm(generated.points[3, 5] - generated.points[3, 2])
    assert width == pytest.approx(1.0, abs=2e-3)


def test_accepts_batched_model_output():
    repo = _to_repo_pixels(_canonical_fixture()).reshape(1, 12, 274)
    seq = adapt_model_output(repo)
    assert seq.num_frames == 12
    with pytest.raises(ValueError):
        adapt_model_output(np.zeros((2, 12, 274), dtype=np.float32))


def test_bin_roundtrip():
    seq = adapt_repo_pose(_to_repo_pixels(_canonical_fixture()), source="real")
    blob = encode_bin(seq, "demo")
    header, arr = decode_bin(blob)
    assert header["sequenceId"] == "demo"
    assert header["T"] == seq.num_frames
    assert header["layout"] == "openpose137"
    assert header["fps"] == 24.0
    assert arr.dtype == np.float32
    assert np.allclose(arr, seq.points)
