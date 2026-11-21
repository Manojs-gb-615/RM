"""
One-command real-data pipeline:
    OpenPose JSON (+ How2Sign CSV)  ->  [T,274] .npy  ->  trained ranker
    ->  real retrieval index  ->  validation metrics

Examples
--------
From official OpenPose clips:
    python ml/build_real_index.py \
        --openpose-json C:/how2sign/openpose_output/json \
        --csv C:/how2sign/how2sign_realigned_val.csv \
        --out-dir data --limit 600 --epochs 40

From already-converted .npy files:
    python ml/build_real_index.py --data-dir data
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.convert_openpose import build_real_from_openpose
from ml.dataset import SignLingoDataset
from ml.ranker import train_ranker, TrainedRetriever
from ml.retrieval import SignLingoRetrieval


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the real How2Sign retrieval index")
    parser.add_argument("--openpose-json", default=None,
                        help="openpose_output/json directory (runs conversion first)")
    parser.add_argument("--csv", default=None, help="How2Sign realigned CSV for the same split")
    parser.add_argument("--limit", type=int, default=600, help="Max clips to convert")
    parser.add_argument("--out-dir", "--data-dir", dest="out_dir", default="data",
                        help="Data directory (poses/ + metadata.csv)")
    parser.add_argument("--index", default="models/retrieval_index.pkl", help="Index output")
    parser.add_argument("--ranker", default="models/ranker.pt", help="Ranker checkpoint output")
    parser.add_argument("--epochs", type=int, default=40, help="Ranker training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--width", type=float, default=1280.0)
    parser.add_argument("--height", type=float, default=720.0)
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing checkpoint")
    args = parser.parse_args()

    if args.openpose_json:
        if not args.csv:
            print("[build] ERROR: --csv is required with --openpose-json", file=sys.stderr)
            return 1
        build_real_from_openpose(
            Path(args.openpose_json), Path(args.csv), Path(args.out_dir),
            limit=args.limit, width=args.width, height=args.height,
        )

    dataset = SignLingoDataset(data_dir=args.out_dir).load()
    samples = dataset.samples_with_pose()
    if not samples:
        print(f"[build] ERROR: no pose samples in {args.out_dir}. "
              "Pass --openpose-json + --csv, or convert first.", file=sys.stderr)
        return 1
    if dataset.is_synthetic:
        print("[build] ERROR: synthetic data is not allowed here. Use ml/train.py --synthetic "
              "for internal tests.", file=sys.stderr)
        return 1

    if args.skip_train and Path(args.ranker).exists():
        print(f"[build] Reusing ranker checkpoint: {args.ranker}")
    else:
        train_ranker(samples, out_path=args.ranker, epochs=args.epochs, seed=args.seed)

    trained = TrainedRetriever.load(args.ranker)
    engine = SignLingoRetrieval(trained=trained)
    engine.trained_checkpoint = args.ranker
    engine.is_synthetic = False
    engine.build_index(samples, trained=trained)
    engine.save(args.index)
    print(f"[build] Real index ready: {args.index} ({engine.num_samples} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
