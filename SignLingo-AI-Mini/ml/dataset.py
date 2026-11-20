"""
SignLingo AI Mini — Dataset Loader
Loads How2Sign pose samples from data/poses/ and data/metadata.csv.
Supports synthetic data for internal testing only.
"""

import os
import csv
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PoseSample:
    sample_id: str
    sentence: str
    video_id: str
    narrative_id: str
    start_time: float
    end_time: float
    pose: Optional[np.ndarray]  # shape [T, 274] (137 keypoints × 2D)
    prev_sample_id: Optional[str] = None
    prev_sentence: Optional[str] = None
    is_synthetic: bool = False


# ── Keypoint layout constants ─────────────────────────────────────────────────

POSE_DIM = 274      # 137 keypoints × 2 coords (x, y)
NUM_KEYPOINTS = 137

# MediaPipe-style skeleton connectivity (keypoint indices for drawing)
BODY_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]
# Left hand: keypoints 25-45 (MediaPipe 21 landmarks)
LHAND_BASE = 25
RHAND_BASE = 46
FACE_BASE  = 67

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

LHAND_CONNECTIONS = [(LHAND_BASE + a, LHAND_BASE + b) for a, b in HAND_CONNECTIONS]
RHAND_CONNECTIONS = [(RHAND_BASE + a, RHAND_BASE + b) for a, b in HAND_CONNECTIONS]


# ── Dataset loader ─────────────────────────────────────────────────────────────

class SignLingoDataset:
    """
    Loads pose samples from data/metadata.csv + data/poses/*.npy.
    Falls back to empty dataset if pose files are unavailable.
    """

    def __init__(self, data_dir: str = "data", pose_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.pose_dir = Path(pose_dir) if pose_dir else self.data_dir / "poses"
        self.metadata_csv = self.data_dir / "metadata.csv"
        self.samples: List[PoseSample] = []
        self._index: Dict[str, PoseSample] = {}
        self._loaded = False
        self.is_synthetic = False

    def load(self) -> "SignLingoDataset":
        """Load metadata CSV and attempt to load pose .npy files."""
        self.is_synthetic = (self.data_dir / "SYNTHETIC_DATASET.txt").exists()
        if not self.metadata_csv.exists():
            print(f"[Dataset] metadata.csv not found at {self.metadata_csv}. Dataset empty.")
            return self
        self.samples = []
        rows = []
        with open(self.metadata_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Build lookup for prev sentences
        id_to_row = {r["sample_id"]: r for r in rows}

        pose_missing = 0
        for r in rows:
            sid = r["sample_id"]
            prev_id = r.get("prev_sample_id", "") or None
            prev_sent = id_to_row.get(prev_id, {}).get("sentence", "") if prev_id else None

            # Try loading pose
            pose = self._load_pose(sid)
            if pose is None:
                pose_missing += 1

            sample = PoseSample(
                sample_id=sid,
                sentence=r["sentence"],
                video_id=r.get("video_id", ""),
                narrative_id=r.get("narrative_id", ""),
                start_time=float(r.get("start_time", 0.0)),
                end_time=float(r.get("end_time", 0.0)),
                pose=pose,
                prev_sample_id=prev_id,
                prev_sentence=prev_sent,
            )
            self.samples.append(sample)
            self._index[sid] = sample

        self._loaded = True
        print(f"[Dataset] Loaded {len(self.samples)} samples. "
              f"Poses available: {len(self.samples) - pose_missing}/{len(self.samples)}.")
        if pose_missing == len(self.samples):
            print("[Dataset] WARNING: No pose files found. Run data setup or download How2Sign poses.")
        return self

    def _load_pose(self, sample_id: str) -> Optional[np.ndarray]:
        """Try to load a pose .npy or .npz file."""
        for ext in [".npy", ".npz"]:
            path = self.pose_dir / f"{sample_id}{ext}"
            if path.exists():
                try:
                    if ext == ".npy":
                        arr = np.load(str(path))
                    else:
                        d = np.load(str(path))
                        arr = d[d.files[0]]
                    if arr.ndim == 2 and arr.shape[1] == POSE_DIM:
                        return arr.astype(np.float32)
                    print(f"[Dataset] Pose {sample_id} has unexpected shape {arr.shape}, skipping.")
                except Exception as e:
                    print(f"[Dataset] Failed to load {path}: {e}")
        return None

    def get_sample(self, sample_id: str) -> Optional[PoseSample]:
        return self._index.get(sample_id)

    def samples_with_pose(self) -> List[PoseSample]:
        return [s for s in self.samples if s.pose is not None]

    def to_dict_list(self) -> List[Dict]:
        """Serialisable list for the API."""
        result = []
        for s in self.samples:
            result.append({
                "sample_id": s.sample_id,
                "sentence": s.sentence,
                "video_id": s.video_id,
                "narrative_id": s.narrative_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "has_pose": s.pose is not None,
                "prev_sample_id": s.prev_sample_id,
                "prev_sentence": s.prev_sentence,
                "frames": int(s.pose.shape[0]) if s.pose is not None else 0,
            })
        return result


# ── Synthetic dataset for internal testing ONLY ───────────────────────────────

def make_synthetic_dataset(n: int = 10, seed: int = 42) -> List[PoseSample]:
    """
    Returns synthetic PoseSample objects with random poses.
    FOR INTERNAL TESTING ONLY. Never display as ASL output.
    """
    rng = np.random.default_rng(seed)
    sentences = [
        "Hello, how are you?",
        "My name is Alex.",
        "I am learning sign language.",
        "This is a test sentence.",
        "The cat sat on the mat.",
        "What time is it?",
        "Thank you very much.",
        "Please repeat that.",
        "I do not understand.",
        "See you tomorrow.",
    ]
    samples = []
    for i in range(n):
        sent = sentences[i % len(sentences)]
        T = rng.integers(20, 60)
        pose = rng.uniform(-1, 1, (T, POSE_DIM)).astype(np.float32)
        prev_id = f"syn_{i-1:03d}" if i > 0 else None
        samples.append(PoseSample(
            sample_id=f"syn_{i:03d}",
            sentence=sent,
            video_id=f"vid_{i:03d}",
            narrative_id="synthetic",
            start_time=float(i * 5),
            end_time=float(i * 5 + 4),
            pose=pose,
            prev_sample_id=prev_id,
            prev_sentence=sentences[(i - 1) % len(sentences)] if i > 0 else None,
            is_synthetic=True,
        ))
    return samples


# ── Metadata CSV generator ────────────────────────────────────────────────────

def create_metadata_template(output_path: str = "data/metadata.csv"):
    """Create an empty metadata CSV template for users to fill in."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "sample_id", "sentence", "video_id", "narrative_id",
        "start_time", "end_time", "prev_sample_id"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        # Example row
        writer.writerow({
            "sample_id": "sample_001",
            "sentence": "Hello, how are you?",
            "video_id": "video_1",
            "narrative_id": "narrative_1",
            "start_time": "0.0",
            "end_time": "2.5",
            "prev_sample_id": "",
        })
    print(f"[Dataset] Template created: {output_path}")


if __name__ == "__main__":
    # Smoke test: load dataset (empty if no files), print stats
    ds = SignLingoDataset().load()
    print(f"Samples: {len(ds.samples)}")
    # Synthetic test
    syn = make_synthetic_dataset(5)
    print(f"Synthetic samples (INTERNAL TEST ONLY): {len(syn)}")
    for s in syn:
        print(f"  {s.sample_id}: '{s.sentence}' | pose shape {s.pose.shape}")
