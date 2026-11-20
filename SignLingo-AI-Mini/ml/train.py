"""
SignLingo AI Mini — Training Script
Builds and saves the retrieval index from How2Sign pose data.
Run this locally after downloading data, or use the Colab notebook.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset import SignLingoDataset, make_synthetic_dataset
from ml.retrieval import SignLingoRetrieval


def _score_sample(engine: SignLingoRetrieval, sample, index_map, mode: str) -> float:
    prev_pose = None
    if sample.prev_sample_id and sample.prev_sample_id in index_map:
        prev_pose = index_map[sample.prev_sample_id].pose
    result = engine.retrieve(
        current_sentence=sample.sentence,
        mode=mode,
        prev_sentence=sample.prev_sentence,
        prev_pose=prev_pose,
        require_pose=False,
    )
    return float(result.score)


def evaluate_retrieval(engine: SignLingoRetrieval, samples, mode: str, holdout_frac: float = 0.25, seed: int = 42) -> dict:
    """
    Honest evaluation with two clearly separated numbers:

    - self_top1: with the full index (query included), how often the top-1 result
      is the query sample itself. This is a pipeline sanity check (~1.0 expected),
      NOT generalisation accuracy.
    - heldout_mean_score: build the index on a train split only, then query the
      held-out samples and report the mean top-1 cosine score. Scores are not
      accuracy; labels for unseen sentences are unavailable.

    If there are too few samples, evaluation is skipped.
    """
    if len(samples) < 4:
        return {"mode": mode, "num_samples": len(samples), "skipped": True,
                "note": "Need at least 4 samples for a split evaluation."}

    index_map = {s.sample_id: s for s in samples}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    n_test = max(1, int(len(samples) * holdout_frac))
    test_positions = set(int(i) for i in order[:n_test])
    train = [s for i, s in enumerate(samples) if i not in test_positions]
    test = [s for i, s in enumerate(samples) if i in test_positions]

    full_engine = SignLingoRetrieval(model_name=engine.model_name)
    full_engine.build_index(samples)
    prev_map = index_map
    self_hits = 0
    for sample in samples:
        prev_pose = prev_map[sample.prev_sample_id].pose if sample.prev_sample_id in prev_map else None
        result = full_engine.retrieve(
            current_sentence=sample.sentence,
            mode=mode,
            prev_sentence=sample.prev_sentence,
            prev_pose=prev_pose,
            require_pose=True,
        )
        if result.matched_sample_id == sample.sample_id:
            self_hits += 1

    train_engine = SignLingoRetrieval(model_name=engine.model_name)
    train_engine.build_index(train)
    heldout_scores = [_score_sample(train_engine, s, index_map, mode) for s in test]

    return {
        "mode": mode,
        "num_samples": len(samples),
        "num_train": len(train),
        "num_test": len(test),
        "self_top1": self_hits / len(samples),
        "heldout_mean_score": float(np.mean(heldout_scores)) if heldout_scores else 0.0,
        "note": (
            "self_top1 is a pipeline sanity check (index contains the query). "
            "heldout_mean_score is a similarity proxy, not accuracy. "
            "Real ASL correctness requires evaluation by qualified signers."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="SignLingo AI Mini — Build retrieval index")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--pose-dir", default=None, help="Pose files directory")
    parser.add_argument("--output", default="models/retrieval_index.pkl", help="Index output path")
    parser.add_argument("--eval", action="store_true", help="Run train/holdout split evaluation")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing (INTERNAL ONLY, not real ASL)")
    args = parser.parse_args()

    if args.synthetic:
        print("=" * 60)
        print("WARNING: Using SYNTHETIC data for internal testing only.")
        print("Synthetic poses are random numbers, NOT real ASL movements.")
        print("=" * 60)
        samples = make_synthetic_dataset(20)
    else:
        ds = SignLingoDataset(data_dir=args.data_dir, pose_dir=args.pose_dir).load()
        samples = ds.samples_with_pose()
        if not samples:
            print("[Train] No pose files found.")
            print("  To get pose data:")
            print("  1. Register at https://how2sign.github.io/")
            print("  2. Download the How2Sign pose files")
            print("  3. Place .npy files in data/poses/ named by sentence_id")
            print("  4. Update data/metadata.csv accordingly")
            print("  OR: Run the Colab notebook (notebooks/train_signlingo.ipynb)")
            sys.exit(1)

    print(f"[Train] Building retrieval index from {len(samples)} samples.")
    engine = SignLingoRetrieval()
    engine.is_synthetic = bool(args.synthetic)
    engine.build_index(samples)
    engine.save(args.output)
    print(f"[Train] Index saved to: {args.output}")

    results = {}
    if args.eval and len(samples) >= 4:
        print("[Train] Running split evaluation (train/holdout)…")
        for mode in ["text_only", "context_aware"]:
            res = evaluate_retrieval(engine, samples, mode)
            results[mode] = res
            print(
                f"  {mode}: self_top1={res['self_top1']:.3f}, "
                f"heldout_mean_score={res['heldout_mean_score']:.3f} "
                f"(n_test={res['num_test']})"
            )
        eval_path = Path(args.output).with_name("eval_results.json")
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Train] Eval results saved to {eval_path}")
    elif args.eval:
        print("[Train] Need ≥4 samples for evaluation.")

    print("[Train] Done.")


if __name__ == "__main__":
    main()
