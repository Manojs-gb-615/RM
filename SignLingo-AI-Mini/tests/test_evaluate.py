import json

import pytest

from ml.evaluate import _format_table, _metrics
from ml.setup_data import build_synthetic


def test_metrics_values():
    m = _metrics([1, 2, 6, None], [1.0, 2.0, 3.0, 4.0])
    assert m["n"] == 4
    assert m["recall@1"] == 0.25
    assert m["recall@5"] == 0.5
    assert m["recall@10"] == 0.75
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 6 + 0) / 4)
    assert m["latency_ms_mean"] == pytest.approx(2.5)


def test_metrics_empty():
    assert _metrics([], []) == {"n": 0}


def test_format_table_contains_rows():
    results = {
        ("full", "text_only"): {"baseline": {"n": 2, "recall@1": 1.0, "recall@5": 1.0,
                                              "recall@10": 1.0, "mrr": 1.0,
                                              "latency_ms_mean": 1.0, "latency_ms_p50": 1.0,
                                              "latency_ms_p95": 1.0}},
    }
    table = _format_table(results)
    assert "baseline" in table
    assert "R@1" in table


def test_evaluate_pipeline_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from ml.dataset import SignLingoDataset
    from ml.evaluate import evaluate
    from ml.ranker import TrainedRetriever, train_ranker
    from ml.retrieval import SignLingoRetrieval

    data_dir = tmp_path / "data"
    build_synthetic(data_dir, n=16)
    samples = SignLingoDataset(data_dir=str(data_dir)).load().samples_with_pose()

    ckpt = tmp_path / "ranker.pt"
    train_ranker(samples, out_path=str(ckpt), epochs=2, batch_size=8, seed=42)
    trained = TrainedRetriever.load(str(ckpt))
    engine = SignLingoRetrieval(trained=trained)
    engine.trained_checkpoint = str(ckpt)
    engine.build_index(samples, trained=trained)
    index = tmp_path / "index.pkl"
    engine.save(str(index))

    out_json = tmp_path / "eval.json"
    out_md = tmp_path / "eval.md"
    payload = evaluate(str(data_dir), str(index), str(ckpt), str(out_json), str(out_md))

    assert out_json.exists() and out_md.exists()
    assert payload["num_samples"] == 16
    assert payload["num_val"] >= 1
    for key in ("full/text_only", "full/context_aware", "context_only/context_aware"):
        assert key in payload["results"]
        for model in ("baseline", "trained"):
            m = payload["results"][key][model]
            assert 0.0 <= m["recall@1"] <= 1.0
            assert 0.0 <= m["mrr"] <= 1.0
            assert m["latency_ms_p50"] >= 0.0
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["num_val"] == payload["num_val"]
