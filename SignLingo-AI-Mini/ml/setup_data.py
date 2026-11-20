"""
SignLingo AI Mini — data setup helper.

Commands
--------
Build data/metadata.csv from a How2Sign realigned CSV + a directory of 2D pose
files (body + hands, [T, 274]):

    python ml/setup_data.py real \
        --csv /path/to/how2sign_realigned_train.csv \
        --pose-dir /path/to/poses \
        --output data/metadata.csv

Create a clearly-labelled synthetic dataset for end-to-end testing ONLY
(NOT real ASL; never present these poses as signing):

    python ml/setup_data.py synthetic --output-dir data/synthetic --n 30

Write an empty metadata template:

    python ml/setup_data.py template --output data/metadata.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

POSE_DIM = 274
METADATA_HEADERS = [
    "sample_id",
    "sentence",
    "video_id",
    "narrative_id",
    "start_time",
    "end_time",
    "prev_sample_id",
]


def _norm_row(row: Dict[str, str]) -> Dict[str, str]:
    return {(k or "").strip().upper(): (v or "").strip() for k, v in row.items()}


def _first(row: Dict[str, str], names: List[str], default: str = "") -> str:
    for name in names:
        if name in row and row[name]:
            return row[name]
    return default


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pose_index(pose_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for ext in ("*.npy", "*.npz"):
        for path in pose_dir.glob(ext):
            index[path.stem] = path
    return index


def _load_pose_shape(path: Path) -> Optional[Tuple[int, int]]:
    try:
        data = np.load(str(path), allow_pickle=False)
        array = data[data.files[0]] if path.suffix == ".npz" else data
        if array.ndim == 3:
            array = array.reshape(array.shape[0], -1)
        if array.ndim != 2:
            return None
        return int(array.shape[0]), int(array.shape[1])
    except Exception:
        return None


_NUM_RANGE = re.compile(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)")


def _match_by_time(video_id: str, start: float, end: float, pose_index: Dict[str, Path]) -> Optional[Path]:
    for stem, path in pose_index.items():
        if video_id and not stem.startswith(video_id):
            continue
        for match in _NUM_RANGE.finditer(stem):
            a, b = float(match.group(1)), float(match.group(2))
            low, high = min(a, b), max(a, b)
            if low <= end and high >= start:
                return path
    return None


def build_real(csv_path: Path, pose_dir: Path, output: Path) -> int:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")

    pose_index = _pose_index(pose_dir)
    if not pose_index:
        raise FileNotFoundError(f"No .npy/.npz pose files found in {pose_dir}")

    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        first_line = handle.readline()
        delimiter = "\t" if "\t" in first_line else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [_norm_row(row) for row in reader]

    records = []
    missing = 0
    bad_shape = 0
    for row in rows:
        sentence = _first(row, ["SENTENCE", "TEXT", "TRANSCRIPT"])
        if not sentence:
            continue
        video_id = _first(row, ["VIDEO_ID", "VIDEO_NAME"])
        start = _to_float(_first(row, ["START_REALISED", "START_REALIGNED", "START_REALI", "START"]))
        end = _to_float(_first(row, ["END_REALISED", "END_REALIGNED", "END_REALI", "END"]))
        sentence_name = _first(row, ["SENTENCE_NAME", "SAMPLE_ID", "ID"])
        narrative_id = _first(row, ["NARRATIVE_ID", "NARRATIVE"]) or video_id

        pose_path = None
        for candidate in (sentence_name, _first(row, ["SENTENCE_ID"])):
            if candidate and candidate in pose_index:
                pose_path = pose_index[candidate]
                break
        if pose_path is None:
            pose_path = _match_by_time(video_id, start, end, pose_index)
        if pose_path is None:
            missing += 1
            continue

        shape = _load_pose_shape(pose_path)
        if shape is None or shape[1] != POSE_DIM:
            bad_shape += 1
            continue

        sample_id = pose_path.stem
        records.append(
            {
                "sample_id": sample_id,
                "sentence": sentence,
                "video_id": video_id or sample_id.split("_")[0],
                "narrative_id": narrative_id,
                "start_time": start,
                "end_time": end,
                "frames": shape[0],
            }
        )

    if not records:
        raise RuntimeError(
            "No valid pose files matched the CSV. Check pose filenames and the "
            "expected [T, 274] shape. See README data setup."
        )

    by_video: Dict[str, List[dict]] = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)
    for group in by_video.values():
        group.sort(key=lambda r: r["start_time"])
        for i, record in enumerate(group):
            record["prev_sample_id"] = group[i - 1]["sample_id"] if i > 0 else ""

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_HEADERS + ["frames"])
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print(f"[setup] Wrote {len(records)} samples to {output}")
    print(f"[setup] Skipped {missing} rows without a matching pose file, {bad_shape} with a bad shape.")
    if bad_shape or missing:
        print("[setup] Tip: pose files must be [T, 274] arrays named by sentence_id "
              "(e.g. --7E8rIUe34_19-6-rgb_front.npy).")
    return len(records)


def build_synthetic(output_dir: Path, n: int, seed: int = 42) -> int:
    rng = np.random.default_rng(seed)
    sentences = [
        "Hello, how are you today?",
        "My name is Sam.",
        "I am learning American Sign Language.",
        "Thank you for your patience.",
        "Please show me the sign for water.",
        "I do not understand.",
        "Can you sign that again?",
        "What is your name?",
        "Nice to meet you.",
        "See you tomorrow.",
        "The weather is nice today.",
        "I need help with this.",
        "Where is the library?",
        "How do you sign this word?",
        "I am a student.",
    ]
    pose_dir = output_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(n):
        frames = int(rng.integers(24, 60))
        t = np.linspace(0.0, 2.0 * np.pi, frames)[:, None]
        freq = rng.uniform(0.5, 2.0, size=(1, POSE_DIM))
        phase = rng.uniform(0.0, np.pi, size=(1, POSE_DIM))
        pose = (np.sin(t * freq + phase) * 0.15 + 0.5).astype(np.float32)
        sample_id = f"syn_{i:03d}"
        np.save(pose_dir / f"{sample_id}.npy", pose)
        records.append(
            {
                "sample_id": sample_id,
                "sentence": sentences[i % len(sentences)],
                "video_id": f"synthetic_video_{i // 5}",
                "narrative_id": "synthetic",
                "start_time": float(i * 4),
                "end_time": float(i * 4 + 3),
                "prev_sample_id": f"syn_{i - 1:03d}" if i > 0 else "",
            }
        )

    metadata = output_dir / "metadata.csv"
    with open(metadata, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_HEADERS)
        writer.writeheader()
        writer.writerows(records)

    (output_dir / "SYNTHETIC_DATASET.txt").write_text(
        "SYNTHETIC DATA — internal testing only.\n"
        "These poses are generated sinusoids, NOT American Sign Language.\n"
        "Never present them as real ASL output.\n",
        encoding="utf-8",
    )
    print(f"[setup] Wrote {len(records)} SYNTHETIC samples to {metadata}")
    print("[setup] WARNING: synthetic poses are NOT real ASL. For testing only.")
    return len(records)


def build_template(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_HEADERS)
        writer.writeheader()
    print(f"[setup] Wrote metadata template to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SignLingo AI Mini — dataset setup")
    sub = parser.add_subparsers(dest="mode", required=True)

    real = sub.add_parser("real", help="Build metadata.csv from How2Sign CSV + pose files")
    real.add_argument("--csv", required=True)
    real.add_argument("--pose-dir", required=True)
    real.add_argument("--output", default="data/metadata.csv")

    syn = sub.add_parser("synthetic", help="Create synthetic data for testing only")
    syn.add_argument("--output-dir", default="data/synthetic")
    syn.add_argument("--n", type=int, default=30)
    syn.add_argument("--seed", type=int, default=42)

    tpl = sub.add_parser("template", help="Write an empty metadata template")
    tpl.add_argument("--output", default="data/metadata.csv")

    args = parser.parse_args()
    try:
        if args.mode == "real":
            build_real(Path(args.csv), Path(args.pose_dir), Path(args.output))
        elif args.mode == "synthetic":
            build_synthetic(Path(args.output_dir), args.n, args.seed)
        else:
            build_template(Path(args.output))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[setup] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
