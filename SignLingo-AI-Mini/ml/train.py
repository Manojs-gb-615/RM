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


def evaluate_retrieval(engine: SignLingoRetrieval, samples, mode: str) -> dict:
    """
    Evaluate retrieval by measuring how often the correct sample is top-1.
    Uses leave-one-out: for each sample, retrieve using all others.
    Returns hit@1 and mean score.
    """
    hits = 0
    scores = []
    for i, sample in enumerate(samples):
        # Build temp index without this sample
        others = [s for j, s in enumerate(samples) if j != i]
        temp_engine = SignLingoRetrieval(model_name=engine.model_name)
        temp_engine.build_index(others)

        result = temp_engine.retrieve(
            current_sentence=sample.sentence,
            mode=mode,
            prev_sentence=sample.prev_sentence,
            prev_pose=samples[i - 1].pose if i > 0 else None,
            require_pose=False,
        )
        scores.append(result.score)
        # Hit if matched sentence is same (or close)
        if result.matched_sentence and result.matched_sentence == sample.sentence:
            hits += 1

    return {
        "mode": mode,
        "num_samples": len(samples),
        "hit_at_1": hits / len(samples) if samples else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "note": "Evaluated with leave-one-out on training set only. Independent test set evaluation pending."
    }


def main():
    parser = argparse.ArgumentParser(description="SignLingo AI Mini — Build retrieval index")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--pose-dir", default=None, help="Pose files directory")
    parser.add_argument("--output", default="models/retrieval_index.pkl", help="Index output path")
    parser.add_argument("--eval", action="store_true", help="Run leave-one-out evaluation")
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
    engine.build_index(samples)
    engine.save(args.output)
    print(f"[Train] Index saved to: {args.output}")

    results = {}
    if args.eval and len(samples) >= 3:
        print("[Train] Running leave-one-out evaluation…")
        for mode in ["text_only", "context_aware"]:
            res = evaluate_retrieval(engine, samples, mode)
            results[mode] = res
            print(f"  {mode}: hit@1={res['hit_at_1']:.3f}, mean_score={res['mean_score']:.3f}")
        # Save results
        Path("models").mkdir(exist_ok=True)
        with open("models/eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("[Train] Eval results saved to models/eval_results.json")
    else:
        if args.eval:
            print("[Train] Need ≥3 samples for evaluation.")

    print("[Train] Done.")


if __name__ == "__main__":
    main()
