import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = json.loads((ROOT / "layouts" / "openpose137.json").read_text(encoding="utf-8"))
SOURCE = json.loads((ROOT / "layouts" / "signlingo_openpose137.json").read_text(encoding="utf-8"))
DEMO = ROOT / "demo_data"


def _demo_arrays():
    manifest = DEMO / "manifest.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return [(entry["id"], np.load(DEMO / entry["file"])) for entry in data["sequences"]]


def test_layout_total_and_regions():
    assert CANONICAL["totalPoints"] == 137
    regions = CANONICAL["regions"]
    assert regions["body"]["start"] == 0 and regions["body"]["count"] == 25
    assert regions["face"]["start"] == 25 and regions["face"]["count"] == 70
    assert regions["leftHand"]["start"] == 95 and regions["leftHand"]["count"] == 21
    assert regions["rightHand"]["start"] == 116 and regions["rightHand"]["count"] == 21
    assert len(regions["face"]["names"]) == 70
    assert len(regions["leftHand"]["names"]) == 21


def test_semantic_indices():
    idx = CANONICAL["indices"]
    assert idx["neck"] == 1
    assert idx["rightShoulder"] == 2 and idx["leftShoulder"] == 5
    assert idx["rightWrist"] == 4 and idx["leftWrist"] == 7
    assert idx["leftHandStart"] == 95 and idx["rightHandStart"] == 116
    assert idx["rightPupil"] == 93 and idx["leftPupil"] == 94
    assert CANONICAL["semantics"]["faceLocal"]["noseTip"] == 30


def test_source_mapping_is_bijective_and_correct():
    mapping = SOURCE["sourceToCanonical"]
    assert len(mapping) == 137
    assert sorted(mapping) == list(range(137))
    # Repository source order: body 0-24, left hand 25-45, right hand 46-66, face 67-136.
    # mapping[canonical] = source index.
    assert mapping[25] == 67    # canonical face start  <- source face start
    assert mapping[95] == 25    # canonical left hand   <- source left hand
    assert mapping[116] == 46   # canonical right hand  <- source right hand
    assert mapping[136] == 66   # canonical right pinky tip
    inverse = SOURCE["canonicalToSource"]
    for src, canon in enumerate(mapping):
        assert inverse[canon] == src


@pytest.mark.parametrize("sequence_id,points", _demo_arrays())
def test_layout_sanity_on_demo_data(sequence_id, points):
    """Catches an accidental layout swap before it reaches the renderer."""
    assert points.ndim == 3 and points.shape[1:] == (137, 2)
    assert np.isfinite(points).all()
    mid = points.shape[0] // 2

    # Shoulder ordering: signer-right (2) is at lower x than signer-left (5).
    assert points[mid, 2, 0] < points[mid, 5, 0]
    # Wrist and the hand root are the same physical point.
    assert np.linalg.norm(points[mid, 7] - points[mid, 95]) < 1e-3
    assert np.linalg.norm(points[mid, 4] - points[mid, 116]) < 1e-3
    # Eyes are above the mouth and inside the head region.
    eye_y = points[mid, 25 + 39, 1]
    mouth_y = points[mid, 25 + 62, 1]
    assert eye_y > mouth_y
    # Nose is between the eyes vertically.
    assert points[mid, 0, 1] < points[mid, 15, 1] + 0.05

    # Finger chains: four points per finger, monotonically shorter last segments.
    for hand_start in (95, 116):
        for finger in range(5):
            base = hand_start + 1 + finger * 4
            chain = points[mid, base:base + 4]
            seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
            assert (seg > 0.01).all(), (sequence_id, base, seg)
            assert seg[-1] <= seg[0] + 0.02, (sequence_id, base, seg)

    # Head width is plausible relative to one shoulder width.
    head_width = np.linalg.norm(points[mid, 25 + 0] - points[mid, 25 + 16])
    assert 0.2 < head_width < 0.8
