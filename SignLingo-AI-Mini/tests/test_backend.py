from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

SRT = """1
00:00:00,000 --> 00:00:02,000
Hello, how are you?

2
00:00:02,000 --> 00:00:04,000
My name is Sam.
"""


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_status_not_ready_without_data():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is False
    assert "is_synthetic" in data


def test_retrieve_not_ready_returns_fallback():
    response = client.post(
        "/api/retrieve",
        json={"current_sentence": "Hello, how are you?", "mode": "text_only"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is False
    assert data["pose"] is None
    assert data["no_match_reason"]


def test_retrieve_rejects_empty_sentence():
    response = client.post("/api/retrieve", json={"current_sentence": "   "})
    assert response.status_code == 422


def test_parse_srt_transcript():
    response = client.post(
        "/api/transcript/parse",
        files={"file": ("sample.srt", SRT, "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["segments"][0]["text"] == "Hello, how are you?"
    assert data["segments"][1]["start_seconds"] == 2.0


def test_parse_rejects_unsupported_extension():
    response = client.post(
        "/api/transcript/parse",
        files={"file": ("sample.txt", "hello", "text/plain")},
    )
    assert response.status_code == 400
