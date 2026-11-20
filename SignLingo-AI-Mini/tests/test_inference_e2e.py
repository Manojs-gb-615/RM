from ml.inference import SignLingoInference
from ml.setup_data import build_synthetic


def test_inference_end_to_end_with_synthetic_data(tmp_path):
    data_dir = tmp_path / "data"
    build_synthetic(data_dir, n=6)

    inference = SignLingoInference(index_path=str(tmp_path / "index.pkl"))
    ready = inference.setup(data_dir=str(data_dir), pose_dir=str(data_dir / "poses"))

    assert ready is True
    status = inference.status()
    assert status["ready"] is True
    assert status["is_synthetic"] is True
    assert status["num_indexed"] >= 1

    result = inference.retrieve_for_sentence("Hello, how are you today?", mode="text_only")
    assert result["ready"] is True
    assert result["is_synthetic"] is True
    assert result["pose"] is not None
    assert len(result["pose"][0]) == 274


def test_inference_not_ready_without_data(tmp_path):
    inference = SignLingoInference(index_path=str(tmp_path / "missing.pkl"))
    ready = inference.setup(data_dir=str(tmp_path / "empty"), pose_dir=None)

    assert ready is False
    result = inference.retrieve_for_sentence("Hello", mode="text_only")
    assert result["ready"] is False
    assert result["pose"] is None
