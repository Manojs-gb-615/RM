"""
Convert official How2Sign OpenPose B-F-H keypoint JSON clips into the
[T, 274] numpy format used by SignLingo AI Mini.

Layout written (137 keypoints x 2 coords):
    0-24    body (OpenPose BODY_25)
    25-45   left hand (21)
    46-66   right hand (21)
    67-136  face (70)

Usage
-----
    python ml/convert_openpose.py \
        --json-dir /path/to/openpose_output/json \
        --csv /path/to/how2sign_realigned_val.csv \
        --out-dir data \
        --limit 600

The output is:
    <out-dir>/poses/<SENTENCE_NAME>.npy
    <out-dir>/metadata.csv
"""

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
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

_FRAME_RE = re.compile(r"_(\d+)_keypoints\.json$")


def _norm_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        (k or "").strip().lstrip("\ufeff").upper(): (v or "").strip()
        for k, v in row.items()
    }


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


def _frame_files(clip_dir: Path) -> List[Path]:
    files = [p for p in clip_dir.glob("*_keypoints.json") if _FRAME_RE.search(p.name)]
    files.sort(key=lambda p: int(_FRAME_RE.search(p.name).group(1)))
    return files


def _pack_keypoints(person: Dict, width: float, height: float, conf: float) -> np.ndarray:
    """Pack one OpenPose person record into 137 xy pairs."""
    out = np.zeros((137, 2), dtype=np.float32)

    def take(key: str, start: int, count: int) -> None:
        flat = person.get(key) or []
        for i in range(min(count, len(flat) // 3)):
            x = float(flat[i * 3])
            y = float(flat[i * 3 + 1])
            c = float(flat[i * 3 + 2])
            if c < conf:
                continue
            out[start + i, 0] = np.clip(x / width, 0.0, 1.0)
            out[start + i, 1] = np.clip(y / height, 0.0, 1.0)

    take("pose_keypoints_2d", 0, 25)
    take("hand_left_keypoints_2d", 25, 21)
    take("hand_right_keypoints_2d", 46, 21)
    take("face_keypoints_2d", 67, 70)
    return out.reshape(-1)


def _pick_person(people: List[Dict]) -> Optional[Dict]:
    if not people:
        return None
    best, best_score = None, -1.0
    for person in people:
        score = 0.0
        for key in ("pose_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
            flat = person.get(key) or []
            score += sum(float(flat[i]) for i in range(2, len(flat), 3))
        if score > best_score:
            best, best_score = person, score
    return best


def convert_clip(clip_dir: Path, width: float, height: float, conf: float) -> Optional[np.ndarray]:
    frames = []
    for path in _frame_files(clip_dir):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        person = _pick_person(data.get("people", []))
        if person is None:
            frames.append(np.zeros(POSE_DIM, dtype=np.float32))
        else:
            frames.append(_pack_keypoints(person, width, height, conf))
    if len(frames) < 2:
        return None
    pose = np.stack(frames).astype(np.float32)
    if not np.any(pose):
        return None
    return pose


def build_real_from_openpose(
    json_dir: Path,
    csv_path: Path,
    out_dir: Path,
    limit: Optional[int] = None,
    width: float = 1280.0,
    height: float = 720.0,
    conf: float = 0.05,
    workers: int = 4,
) -> int:
    if not json_dir.is_dir():
        raise FileNotFoundError(f"OpenPose JSON directory not found: {json_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"How2Sign CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        first = handle.readline()
        delimiter = "\t" if "\t" in first else ","
        handle.seek(0)
        rows = [_norm_row(r) for r in csv.DictReader(handle, delimiter=delimiter)]

    pose_dir = out_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)

    # Resume: reuse already-converted .npy files.
    def make_meta(row: Dict[str, str], sentence_name: str, sentence: str, frames: int = 0) -> Dict:
        video_id = _first(row, ["VIDEO_ID", "VIDEO_NAME"]) or sentence_name.split("_")[0]
        return {
            "sample_id": sentence_name,
            "sentence": sentence,
            "video_id": video_id,
            "narrative_id": video_id,
            "start_time": _to_float(_first(row, ["START_REALIGNED", "START_REALISED", "START"])),
            "end_time": _to_float(_first(row, ["END_REALIGNED", "END_REALISED", "END"])),
            "frames": frames,
        }

    records = []
    pending: List[Tuple[Dict, Path]] = []
    for row in rows:
        if limit is not None and len(records) + len(pending) >= limit:
            break
        sentence = _first(row, ["SENTENCE", "TEXT", "TRANSCRIPT"])
        sentence_name = _first(row, ["SENTENCE_NAME", "SAMPLE_ID", "ID"])
        if not sentence or not sentence_name:
            continue
        target = pose_dir / f"{sentence_name}.npy"
        if target.exists():
            try:
                frames = int(np.load(target, mmap_mode="r").shape[0])
            except Exception:
                continue
            records.append(make_meta(row, sentence_name, sentence, frames))
            continue
        clip_dir = json_dir / sentence_name
        if not clip_dir.is_dir():
            continue
        pending.append((make_meta(row, sentence_name, sentence), clip_dir))
        if len(pending) % 100 == 0:
            print(f"[convert] queued {len(pending)} clips…", flush=True)

    def job(item: Tuple[Dict, Path]):
        meta, clip_dir = item
        return meta, convert_clip(clip_dir, width, height, conf)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for meta, pose in pool.map(job, pending):
            done += 1
            if pose is None:
                continue
            np.save(pose_dir / f"{meta['sample_id']}.npy", pose)
            meta["frames"] = int(pose.shape[0])
            records.append(meta)
            if done % 25 == 0:
                print(f"[convert] {done}/{len(pending)} pending clips processed, "
                      f"{len(records)} total…", flush=True)

    if not records:
        raise RuntimeError("No clips converted.")

    if not records:
        raise RuntimeError(
            "No clips converted. Check that --json-dir points at openpose_output/json "
            "and that SENTENCE_NAME values match folder names."
        )

    by_video: Dict[str, List[dict]] = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)
    for group in by_video.values():
        group.sort(key=lambda r: r["start_time"])
        for i, record in enumerate(group):
            record["prev_sample_id"] = group[i - 1]["sample_id"] if i > 0 else ""

    metadata = out_dir / "metadata.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_HEADERS + ["frames"])
        writer.writeheader()
        writer.writerows(records)

    (out_dir / "DATASET_SOURCE.txt").write_text(
        "Real How2Sign 2D keypoints (B-F-H / OpenPose).\n"
        "Source: https://how2sign.github.io/ (Duarte et al., CVPR 2021).\n"
        "License: CC BY-NC 4.0 — research use only, non-commercial.\n"
        "Converted by ml/convert_openpose.py.\n",
        encoding="utf-8",
    )
    print(f"[convert] Wrote {len(records)} clips to {pose_dir}")
    print(f"[convert] Wrote metadata to {metadata}")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert How2Sign OpenPose JSON to [T,274] .npy")
    parser.add_argument("--json-dir", required=True, help="openpose_output/json directory")
    parser.add_argument("--csv", required=True, help="How2Sign realigned CSV for the same split")
    parser.add_argument("--out-dir", default="data", help="Output directory (poses/ + metadata.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Max clips to convert")
    parser.add_argument("--width", type=float, default=1280.0, help="Source frame width in px")
    parser.add_argument("--height", type=float, default=720.0, help="Source frame height in px")
    parser.add_argument("--conf", type=float, default=0.05, help="Min OpenPose confidence")
    parser.add_argument("--workers", type=int, default=4, help="Parallel conversion workers")
    args = parser.parse_args()
    try:
        build_real_from_openpose(
            Path(args.json_dir), Path(args.csv), Path(args.out_dir),
            limit=args.limit, width=args.width, height=args.height, conf=args.conf,
            workers=args.workers,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[convert] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
