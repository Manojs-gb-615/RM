import pytest
from fastapi.testclient import TestClient

from backend.main import app, inference
from ml.setup_data import build_synthetic

SRT = """1
00:00:00,000 --> 00:00:02,000
Hello, how are you today?

2
00:00:02,000 --> 00:00:04,000
My name is Sam.

3
00:00:04,000 --> 00:00:06,000
I am learning American Sign Language.
"""


@pytest.fixture
def synthetic_client(tmp_path):
    data_dir = tmp_path / "data"
    build_synthetic(data_dir, n=9)

    old_index_path = inference.index_path
    inference.index_path = str(tmp_path / "index.pkl")
    inference.setup(data_dir=str(data_dir), pose_dir=str(data_dir / "poses"))
    client = TestClient(app)
    try:
        yield client
    finally:
        inference.index_path = old_index_path
        inference._engine = None
        inference._dataset = None
        inference._ready = False


def test_status_ready_with_data(synthetic_client):
    status = synthetic_client.get("/api/status").json()
    assert status["ready"] is True
    assert status["is_synthetic"] is True
    assert status["num_indexed"] >= 1


def test_retrieve_all_returns_pose_per_segment(synthetic_client):
    response = synthetic_client.post(
        "/api/transcript/retrieve_all",
        files={"transcript_file": ("sample.srt", SRT, "text/plain")},
        data={"mode": "context_aware"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    first = data["results"][0]["retrieval"]
    assert first["pose"] is not None
    assert len(first["pose"][0]) == 274
    assert first["is_synthetic"] is True
    # later segments carry previous-sentence context
    assert data["results"][1]["retrieval"]["mode"] == "context_aware"


def test_compare_returns_both_modes(synthetic_client):
    response = synthetic_client.post(
        "/api/compare",
        json={"current_sentence": "My name is Sam.", "prev_sentence": "Hello, how are you today?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text_only"] is not None
    assert data["context_aware"] is not None
    assert "same_result" in data


def test_retrieve_no_match_fallback(synthetic_client):
    response = synthetic_client.post(
        "/api/retrieve",
        json={"current_sentence": "quantum chromodynamics lattice gauge theory", "mode": "text_only"},
    )
    assert response.status_code == 200
    data = response.json()
    if data["matched_sample_id"] is None:
        assert data["pose"] is None
        assert data["no_match_reason"]
