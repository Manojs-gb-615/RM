"""API integration and unit tests for ContextSign AI backend."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root_redirect():
    """Test that GET / redirects to /docs."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [302, 307]
    assert response.headers["location"] == "/docs"


def test_health_check():
    """Test GET /api/health returns operational status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "model_ready" in data
    assert "cuda_available" in data
    assert "device" in data
    # In initial state without checkpoint, model_ready must be False
    assert data["model_ready"] is False


def test_model_status_not_ready():
    """Test GET /api/model/status confirms model is not ready without checkpoint."""
    response = client.get("/api/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is False
    assert data["checkpoint_path"] is None
    assert "cuda_available" in data


def test_generate_returns_503_when_model_not_ready():
    """Test POST /api/generate gracefully returns 503 ModelNotReady when no checkpoint is loaded."""
    payload = {
        "text": "Hello, thank you for your help today.",
        "context_type": "none",
        "max_frames": 100,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 503
    data = response.json()
    assert data["error"] == "ModelNotReady"
    assert "detail" in data
    assert "suggestion" in data
    assert "checkpoint" in data["detail"].lower()


def test_generate_validation_error():
    """Test POST /api/generate returns 422 on invalid payload (e.g., text exceeding max length)."""
    # Exceeding 500 characters
    long_text = "sign language " * 50
    payload = {
        "text": long_text,
        "max_frames": 100,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "ValidationError"
    assert "detail" in data


def test_generate_validation_empty_text():
    """Test POST /api/generate returns 422 on empty text."""
    payload = {
        "text": "",
        "max_frames": 100,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "ValidationError"


def test_compare_no_checkpoints():
    """Test POST /api/compare handles missing checkpoints gracefully with null outputs."""
    payload = {
        "text": "Can you please repeat that more slowly?",
        "max_frames": 120,
    }
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["baseline"] is None
    assert data["context_aware"] is None
    assert data["metrics"] is not None
    assert data["metrics"]["baseline_checkpoint_found"] is False
    assert data["metrics"]["context_checkpoint_found"] is False


def test_dataset_samples():
    """Test GET /api/dataset/samples returns a valid paginated response."""
    response = client.get("/api/dataset/samples?split=train&limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert "total" in data
    assert data["split"] == "train"
    assert isinstance(data["samples"], list)
    assert isinstance(data["total"], int)


def test_dataset_sample_not_found():
    """Test GET /api/dataset/samples/{id} returns 404 for an invalid sample ID."""
    response = client.get("/api/dataset/samples/nonexistent_sample_xyz_99999")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_experiments_list():
    """Test GET /api/experiments returns all E1-E9 experiments with correct structure."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    exp_ids = [e["id"] for e in data["experiments"]]
    # Must contain E1 through E9
    for i in range(1, 10):
        assert f"E{i}" in exp_ids

    # Check status values are valid
    valid_statuses = {"IMPLEMENTED", "TESTED", "TRAINED", "EVALUATED", "PENDING"}
    for e in data["experiments"]:
        assert e["status"] in valid_statuses
        assert "name" in e
        assert "description" in e


def test_metrics_not_evaluated():
    """Test GET /api/metrics returns not_evaluated when no evaluation results exist."""
    response = client.get("/api/metrics?experiment_id=e1")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == "E1"
    assert "metrics" in data
    assert data["data_type"] in ["not_evaluated", "synthetic_smoke_test", "real_how2sign"]
    assert "timestamp" in data


def test_architecture():
    """Test GET /api/architecture returns system specifications and component list."""
    response = client.get("/api/architecture")
    assert response.status_code == 200
    data = response.json()
    assert "components" in data
    assert "data_flow" in data
    assert "model_config" in data

    component_names = [c["name"] for c in data["components"]]
    assert "TextEncoder" in component_names
    assert "PoseContextEncoder" in component_names
    assert "CrossAttentionFusion" in component_names
    assert "CVAE" in component_names
    assert "TransformerDecoder" in component_names

    assert len(data["data_flow"]) >= 5
    assert data["model_config"]["hidden_dim"] == 256
    assert data["model_config"]["pose_dim"] == 274


@pytest.mark.asyncio
async def test_async_client_health():
    """Test async client interaction using httpx if available."""
    try:
        import httpx
        from httpx import ASGITransport

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            res = await ac.get("/api/health")
            assert res.status_code == 200
            assert res.json()["status"] == "healthy"
    except ImportError:
        pytest.skip("httpx not installed or ASGITransport not supported")
