"""
Automated retrieval evaluation for SignLingo AI Mini.

Reports Recall@1/5/10, MRR and per-query latency for:
    - baseline fixed weighted cosine ranker
    - trained dual-encoder ranker
on the validation split (from the ranker checkpoint).

Two protocols are reported:
    full      : the query sentence is present in the index (standard retrieval)
    ctx-only  : the current sentence is hidden; retrieval must use the previous
                sentence + previous pose (non-trivial context ablation)

Usage:
    python ml/evaluate.py --data-dir data --index models/retrieval_index.pkl \
        --ranker models/ranker.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset import SignLingoDataset
from ml.retrieval import SignLingoRetrieval


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _ranks_for_queries(
    engine: SignLingoRetrieval,
    samples,
    val_ids: Sequence[str],
    mode: str,
    include_current: bool,
) -> Tuple[List[Optional[int]], List[float]]:
    by_id = {s.sample_id: s for s in samples}
    relevant_sets: Dict[str, set] = {}
    for s in samples:
        relevant_sets.setdefault(_norm(s.sentence), set()).add(s.sample_id)

    ranks: List[Optional[int]] = []
    latencies: List[float] = []
    for sample_id in val_ids:
        sample = by_id.get(sample_id)
        if sample is None or sample.pose is None:
            continue
        prev = by_id.get(sample.prev_sample_id) if sample.prev_sample_id else None
        t0 = time.perf_counter()
        ranked = engine.score_all(
            current_sentence=sample.sentence,
            mode=mode,
            prev_sentence=sample.prev_sentence or (prev.sentence if prev else None),
            prev_pose=prev.pose if prev else None,
            require_pose=True,
            include_current=include_current,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        relevant = relevant_sets[_norm(sample.sentence)]
        rank = None
        for i, (_, entry) in enumerate(ranked):
            if entry["sample_id"] in relevant:
                rank = i + 1
                break
        ranks.append(rank)
    return ranks, latencies


def _metrics(ranks: List[Optional[int]], latencies: List[float]) -> Dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    def recall(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    lat = np.array(latencies, dtype=np.float64)
    return {
        "n": n,
        "recall@1": recall(1),
        "recall@5": recall(5),
        "recall@10": recall(10),
        "mrr": float(mrr),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
    }


def _format_table(results: Dict) -> str:
    lines = [
        "| Protocol | Mode | Model | R@1 | R@5 | R@10 | MRR | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    labels = {
        ("full", "text_only"): "full",
        ("full", "context_aware"): "full",
        ("context_only", "context_aware"): "ctx-only",
    }
    for (protocol, mode), per_model in results.items():
        for model_name, m in per_model.items():
            if not m or "n" not in m or m["n"] == 0:
                continue
            lines.append(
                f"| {labels.get((protocol, mode), protocol)} | {mode} | {model_name} | "
                f"{m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
                f"{m['mrr']:.3f} | {m['latency_ms_p50']:.1f} | {m['latency_ms_p95']:.1f} |"
            )
    return "\n".join(lines)


def evaluate(
    data_dir: str = "data",
    index_path: str = "models/retrieval_index.pkl",
    ranker_path: str = "models/ranker.pt",
    output_json: str = "models/eval_results.json",
    output_md: str = "docs/eval_results.md",
) -> Dict:
    dataset = SignLingoDataset(data_dir=data_dir).load()
    samples = [s for s in dataset.samples if s.pose is not None]
    if len(samples) < 8:
        raise RuntimeError("Need at least 8 pose samples for evaluation.")

    # Validation ids from the checkpoint (same sample ordering used in training).
    import torch
    checkpoint = torch.load(ranker_path, map_location="cpu", weights_only=False)
    val_indices = [i for i in checkpoint.get("val_indices", []) if i < len(samples)]
    if not val_indices:
        rng = np.random.default_rng(int(checkpoint.get("seed", 42)))
        order = rng.permutation(len(samples))
        val_indices = sorted(int(i) for i in order[: max(1, int(len(samples) * 0.2))])
    val_ids = [samples[i].sample_id for i in val_indices]

    baseline = SignLingoRetrieval().build_index(samples)
    trained = SignLingoRetrieval.load(index_path)
    if not trained.has_trained:
        raise RuntimeError(
            f"Index {index_path} has no trained ranker. Build it with "
            "python ml/train.py --train-ranker"
        )

    protocols = [
        ("full", "text_only", True),
        ("full", "context_aware", True),
        ("context_only", "context_aware", False),
    ]
    results: Dict = {}
    for protocol, mode, include_current in protocols:
        key = (protocol, mode)
        results[key] = {}
        for model_name, engine in (("baseline", baseline), ("trained", trained)):
            ranks, lat = _ranks_for_queries(engine, samples, val_ids, mode, include_current)
            results[key][model_name] = _metrics(ranks, lat)

    payload = {
        "data_dir": data_dir,
        "index": index_path,
        "ranker": ranker_path,
        "num_samples": len(samples),
        "num_val": len(val_ids),
        "note": (
            "full protocol includes the query caption in the index, so exact-caption "
            "retrieval is near-trivial. ctx-only hides the current sentence and is the "
            "non-trivial context ablation. Real ASL correctness still requires signer study."
        ),
        "results": {
            f"{protocol}/{mode}": per_model
            for (protocol, mode), per_model in results.items()
        },
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    table = _format_table(results)
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text(
        "# SignLingo AI Mini — Retrieval Evaluation\n\n"
        f"- Samples: {len(samples)} (val n={len(val_ids)})\n"
        f"- Index: `{index_path}`\n- Ranker: `{ranker_path}`\n\n"
        f"{table}\n\n"
        "`full` includes the query caption in the index (exact-caption retrieval is "
        "near-trivial). `ctx-only` hides the current sentence, so it measures whether "
        "context + previous pose carry signal. These are retrieval metrics, not ASL "
        "correctness; human signer evaluation is pending.\n",
        encoding="utf-8",
    )
    print(table)
    print(f"\n[evaluate] Wrote {output_json} and {output_md}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs trained retrieval")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index", default="models/retrieval_index.pkl")
    parser.add_argument("--ranker", default="models/ranker.pt")
    parser.add_argument("--output-json", default="models/eval_results.json")
    parser.add_argument("--output-md", default="docs/eval_results.md")
    args = parser.parse_args()
    try:
        evaluate(args.data_dir, args.index, args.ranker, args.output_json, args.output_md)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[evaluate] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
