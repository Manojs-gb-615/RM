"""Prepare demo pose sequences for the avatar renderer.

Search order for real data:
  1. --pose-dir (directory of [T,274] .npy files in the SignLingo OpenPose layout)
  2. SignLingo-AI-Mini/data/real/poses
  3. data/real/poses
  4. $HOW2SIGN_POSE_DIR

If a metadata CSV is available it supplies captions and sentence boundaries
(one sample per sentence). If no real poses are found, four clearly-labelled
synthetic motions plus an 8-second synthetic session are written instead.

Every sequence is converted to canonical space by
backend/services/canonical_pose.py and stored as float32 [T,137,2] .npy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.canonical_pose import adapt_repo_pose  # noqa: E402
from backend.services.synthetic_pose import SENTENCE_TEXTS, build_all  # noqa: E402

SOURCE_LAYOUT = "signlingo_openpose137"
DISCLAIMER = (
    "Ground-truth or synthetic demo pose data. Synthetic sequences are NOT ASL. "
    "No model output and no claim of signing accuracy."
)


def _find_pose_dir(explicit: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        REPO_ROOT / "SignLingo-AI-Mini" / "data" / "real" / "poses",
        REPO_ROOT / "data" / "real" / "poses",
        REPO_ROOT / "data" / "poses",
    ]
    env = os.environ.get("HOW2SIGN_POSE_DIR")
    if env:
        candidates.append(Path(env))
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.npy")):
            return candidate
    return None


def _find_metadata(explicit: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        REPO_ROOT / "SignLingo-AI-Mini" / "data" / "real" / "metadata.csv",
        REPO_ROOT / "data" / "real" / "metadata.csv",
        REPO_ROOT / "data" / "metadata.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_metadata(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("sample_id")}


def _save(out_dir: Path, sequence_id: str, points: np.ndarray) -> str:
    filename = f"{sequence_id}.npy"
    np.save(out_dir / filename, np.ascontiguousarray(points, dtype=np.float32))
    return filename


def _synthetic_sequences(out_dir: Path, fps: float, sequences: List[Dict]) -> None:
    for kind, (points, sentences) in build_all(fps).items():
        sequence_id = f"synthetic_{kind}"
        filename = _save(out_dir, sequence_id, points)
        title = {
            "arm_raise": "Synthetic — arm raise",
            "handshape_cycle": "Synthetic — handshape cycle",
            "head_nod": "Synthetic — head nod",
            "brow_raise": "Synthetic — brow raise",
            "session": "Synthetic — combined session (4 sentences)",
        }[kind]
        sequences.append({
            "id": sequence_id,
            "title": title,
            "file": filename,
            "frames": int(points.shape[0]),
            "fps": float(fps),
            "source": "synthetic",
            "synthetic": True,
            "layout": "openpose137",
            "normalization": "neck_origin_shoulder_width",
            "sentences": sentences,
            "text": " / ".join(str(s["text"]) for s in sentences),
            "warning": "Synthetic test motion — not American Sign Language.",
        })


def _real_sequences(out_dir: Path, pose_dir: Path, metadata: Dict[str, Dict[str, str]],
                    limit: int, fps: float, sequences: List[Dict]) -> None:
    files = sorted(pose_dir.glob("*.npy"))
    # Prefer files that have metadata, keep chronological order.
    files.sort(key=lambda p: (
        float(metadata.get(p.stem, {}).get("start_time") or 0.0), p.stem
    ))
    made = 0
    for path in files:
        if made >= limit:
            break
        try:
            raw = np.load(path)
        except Exception:
            continue
        if raw.ndim != 2 or raw.shape[1] != 274:
            continue
        row = metadata.get(path.stem, {})
        try:
            seq = adapt_repo_pose(
                raw,
                source_layout=SOURCE_LAYOUT,
                source="real",
                fps=fps,
            )
        except Exception as exc:  # noqa: BLE001 - report and skip bad file
            print(f"[prepare] skipping {path.name}: {exc}")
            continue
        sequence_id = f"real_{path.stem}"
        filename = _save(out_dir, sequence_id, seq.points)
        sequences.append({
            "id": sequence_id,
            "title": f"How2Sign — {path.stem}",
            "file": filename,
            "frames": seq.num_frames,
            "fps": float(fps),
            "source": "real",
            "synthetic": False,
            "layout": "openpose137",
            "normalization": "neck_origin_shoulder_width",
            "sentences": [{
                "index": 0,
                "startFrame": 0,
                "endFrame": seq.num_frames,
                "text": row.get("sentence", ""),
            }],
            "text": row.get("sentence", ""),
            "sourceSampleId": path.stem,
            "warning": "Ground-truth OpenPose keypoints. Not reviewed by ASL signers.",
        })
        made += 1
    print(f"[prepare] adapted {made} real clip(s) from {pose_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare canonical demo pose sequences")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "demo_data"))
    parser.add_argument("--pose-dir", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--fps", type=float, default=24.0, help="Fallback fps when metadata has none")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"[prepare] {manifest_path} exists; use --force to rebuild.")
        return 0

    sequences: List[Dict] = []
    pose_dir = _find_pose_dir(args.pose_dir)
    metadata = _read_metadata(_find_metadata(args.metadata))
    real_found = pose_dir is not None
    if real_found:
        _real_sequences(out_dir, pose_dir, metadata, args.limit, args.fps, sequences)
    else:
        print("[prepare] No real pose .npy files found; writing synthetic demo data.")

    # Synthetic sequences are always included: they exercise handshape, head,
    # face and sentence-boundary code paths even when real data is present.
    _synthetic_sequences(out_dir, args.fps, sequences)

    # Remove stale files that are no longer in the manifest.
    keep = {s["file"] for s in sequences}
    for path in out_dir.glob("*.npy"):
        if path.name not in keep:
            path.unlink()

    manifest = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fps": args.fps,
        "layout": "openpose137",
        "normalization": "neck_origin_shoulder_width",
        "coordinateSpace": "origin=neck, unit=median shoulder width, +x signer-left, +y up, +z camera",
        "sourceLayout": SOURCE_LAYOUT if real_found else "openpose137",
        "realDataAvailable": real_found,
        "defaultSequenceId": sequences[0]["id"] if sequences else None,
        "sessionSequenceId": "synthetic_session",
        "disclaimer": DISCLAIMER,
        "sequences": sequences,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (out_dir / "SOURCE.md").write_text(
        "# Demo pose data\n\n"
        f"- Generated: {manifest['generatedAt']}\n"
        f"- Real keypoints available: {real_found}\n"
        f"- Source layout for real data: {SOURCE_LAYOUT}\n"
        "- Canonical layout: `layouts/openpose137.json`\n"
        "- Coordinate space: neck origin, median shoulder-width unit, +x signer-left, +y up.\n\n"
        "Synthetic sequences are hand-authored test motions (arm raise, handshape cycle,\n"
        "head nod, brow raise). They are **not American Sign Language** and must not be\n"
        "presented as signing output.\n",
        encoding="utf-8",
    )
    print(f"[prepare] wrote {len(sequences)} sequence(s) + manifest to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
