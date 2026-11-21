import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml.ranker import (
    DualEncoder,
    TrainedRetriever,
    info_nce,
    pose_features,
    train_on_features,
)
from ml.dataset import make_synthetic_dataset
from ml.retrieval import SignLingoRetrieval


def test_pose_features_shape_and_empty():
    pose = np.zeros((10, 274), dtype=np.float32)
    assert pose_features(pose).shape == (548,)
    assert pose_features(None).shape == (548,)


def test_info_nce_backward():
    model = DualEncoder(hidden=32, out_dim=16)
    text = torch.randn(6, 384)
    pose = torch.randn(6, 548)
    loss = info_nce(model.encode_text(text), model.encode_pose(pose), model.logit_scale)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.text_tower.net[1].weight.grad is not None
    assert model.pose_tower.net[1].weight.grad is not None


def test_train_on_features_saves_checkpoint_and_log(tmp_path):
    rng = np.random.default_rng(0)
    n = 24
    text = rng.normal(size=(n, 384)).astype(np.float32)
    pose = rng.normal(size=(n, 548)).astype(np.float32)
    ckpt = tmp_path / "ranker.pt"
    log = tmp_path / "log.csv"
    meta = train_on_features(
        text, pose, str(ckpt), epochs=3, batch_size=8, seed=7, log_path=str(log)
    )
    assert ckpt.exists() and log.exists()
    assert meta["num_train"] + meta["num_val"] == n
    assert len(meta["loss_log"]) == 3
    loaded = TrainedRetriever.load(str(ckpt))
    out = loaded.encode_text(text[:2])
    assert out.shape == (2, 128)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)


def test_trained_index_roundtrip(tmp_path):
    samples = make_synthetic_dataset(12)
    rng = np.random.default_rng(1)
    text = rng.normal(size=(len(samples), 384)).astype(np.float32)
    pose = np.stack([pose_features(s.pose) for s in samples])
    ckpt = tmp_path / "ranker.pt"
    train_on_features(text, pose, str(ckpt), epochs=2, batch_size=6, seed=3)
    trained = TrainedRetriever.load(str(ckpt))

    engine = SignLingoRetrieval(trained=trained)
    engine.trained_checkpoint = str(ckpt)
    engine.build_index(samples, trained=trained)
    assert engine.has_trained

    result = engine.retrieve(samples[0].sentence, mode="text_only")
    assert result.pose is not None
    assert result.matched_sample_id is not None

    path = tmp_path / "index.pkl"
    engine.save(str(path))
    loaded = SignLingoRetrieval.load(str(path))
    assert loaded.has_trained is True
    assert loaded.retrieve(samples[1].sentence, mode="context_aware").pose is not None
