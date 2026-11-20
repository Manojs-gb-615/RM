import numpy as np

from ml.dataset import make_synthetic_dataset, POSE_DIM
from ml.retrieval import SignLingoRetrieval
from ml.train import evaluate_retrieval


def test_synthetic_dataset_shapes():
    samples = make_synthetic_dataset(8)
    assert len(samples) == 8
    for sample in samples:
        assert sample.pose is not None
        assert sample.pose.ndim == 2
        assert sample.pose.shape[1] == POSE_DIM


def test_text_only_retrieves_self():
    samples = make_synthetic_dataset(10)
    engine = SignLingoRetrieval()
    engine.build_index(samples)

    result = engine.retrieve(samples[0].sentence, mode="text_only")
    assert result.matched_sample_id == samples[0].sample_id
    assert result.pose is not None
    assert result.pose.shape[1] == POSE_DIM


def test_context_aware_returns_pose():
    samples = make_synthetic_dataset(10)
    engine = SignLingoRetrieval()
    engine.build_index(samples)

    result = engine.retrieve(
        current_sentence=samples[1].sentence,
        mode="context_aware",
        prev_sentence=samples[0].sentence,
        prev_pose=samples[0].pose,
    )
    assert result.mode == "context_aware"
    assert result.pose is not None


def test_save_load_preserves_synthetic_flag(tmp_path):
    samples = make_synthetic_dataset(6)
    engine = SignLingoRetrieval()
    engine.is_synthetic = True
    engine.build_index(samples)

    path = tmp_path / "index.pkl"
    engine.save(str(path))
    loaded = SignLingoRetrieval.load(str(path))

    assert loaded.is_synthetic is True
    assert loaded.num_samples == len(samples)


def test_empty_index_returns_no_match():
    engine = SignLingoRetrieval()
    engine.build_index([])
    result = engine.retrieve("Hello", mode="text_only")
    assert result.matched_sample_id is None
    assert result.pose is None
    assert result.no_match_reason


def test_evaluation_metrics_are_bounded():
    samples = make_synthetic_dataset(12)
    engine = SignLingoRetrieval()
    engine.build_index(samples)

    res = evaluate_retrieval(engine, samples, "text_only")
    assert 0.0 <= res["self_top1"] <= 1.0
    assert -1.0 <= res["heldout_mean_score"] <= 1.0
    assert res["num_test"] >= 1


def test_summarize_pose_feature_size():
    from ml.retrieval import summarize_pose

    pose = np.zeros((10, POSE_DIM), dtype=np.float32)
    feat = summarize_pose(pose)
    assert feat.shape == (64,)
