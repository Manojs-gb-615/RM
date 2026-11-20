"""
SignLingo AI Mini — Context-Aware ASL Pose Retrieval
Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings.
Two retrieval modes:
  - text_only: match current sentence only
  - context_aware: match current sentence + previous sentence + previous pose features
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("[Retrieval] sentence-transformers not installed. Run: pip install sentence-transformers")

from ml.dataset import PoseSample, POSE_DIM

_MODEL_CACHE: Dict[str, "SentenceTransformer"] = {}


# ── Retrieval result ───────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    matched_sample_id: Optional[str]
    matched_sentence: Optional[str]
    score: float
    pose: Optional[np.ndarray]   # shape [T, 274] or None
    no_match_reason: Optional[str] = None  # why no pose was returned
    mode: str = "unknown"


NO_POSE_RESULT = RetrievalResult(
    matched_sample_id=None,
    matched_sentence=None,
    score=0.0,
    pose=None,
    no_match_reason="No reliable ASL pose available for this sentence.",
    mode="none",
)


# ── Pose feature summary ──────────────────────────────────────────────────────

def summarize_pose(pose: np.ndarray, dim: int = 32) -> np.ndarray:
    """
    Compress a pose sequence [T, 274] to a fixed-size feature vector.
    Uses temporal mean + std of the first `dim` keypoint coords.
    """
    if pose is None or pose.shape[0] == 0:
        return np.zeros(dim * 2, dtype=np.float32)
    subset = pose[:, : dim]      # [T, dim]
    return np.concatenate([subset.mean(axis=0), subset.std(axis=0)]).astype(np.float32)


# ── Retrieval engine ──────────────────────────────────────────────────────────

class SignLingoRetrieval:
    """
    Builds an index of sentence embeddings (+ optional pose features)
    and retrieves the best matching sample for a query sentence.
    """

    # Minimum cosine similarity score to return a match
    MATCH_THRESHOLD = 0.30
    # Weight on pose feature similarity when context is available
    POSE_WEIGHT = 0.25
    TEXT_WEIGHT = 0.75

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._sbert: Optional[SentenceTransformer] = None
        self._index: List[Dict] = []   # [{sample_id, sentence, text_emb, pose_feat}]
        self._built = False
        self.is_synthetic = False

    # ── Lazy load sentence-transformer ────────────────────────────────────────

    def _get_model(self) -> "SentenceTransformer":
        if self._sbert is None:
            if not HAS_SBERT:
                raise RuntimeError("Install sentence-transformers: pip install sentence-transformers")
            if self.model_name not in _MODEL_CACHE:
                print(f"[Retrieval] Loading sentence-transformer: {self.model_name}")
                _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
            self._sbert = _MODEL_CACHE[self.model_name]
        return self._sbert

    # ── Build index ────────────────────────────────────────────────────────────

    def build_index(self, samples: List[PoseSample]) -> "SignLingoRetrieval":
        """Encode all sample sentences and cache pose features."""
        if not samples:
            print("[Retrieval] No samples to index.")
            self._built = True
            return self
        model = self._get_model()
        sentences = [s.sentence for s in samples]
        print(f"[Retrieval] Encoding {len(sentences)} sentences…")
        embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms

        self._index = []
        for s, emb in zip(samples, embeddings):
            self._index.append({
                "sample_id": s.sample_id,
                "sentence": s.sentence,
                "text_emb": emb,
                "pose_feat": summarize_pose(s.pose) if s.pose is not None else None,
                "pose": s.pose,
                "has_pose": s.pose is not None,
            })
        self._built = True
        print(f"[Retrieval] Index built: {len(self._index)} entries, "
              f"{sum(1 for e in self._index if e['has_pose'])} with pose.")
        return self

    # ── Retrieve ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        current_sentence: str,
        mode: str = "text_only",           # "text_only" | "context_aware"
        prev_sentence: Optional[str] = None,
        prev_pose: Optional[np.ndarray] = None,
        require_pose: bool = True,
    ) -> RetrievalResult:
        """
        Find the best matching sample for current_sentence.
        mode='text_only':      cosine similarity on current sentence embedding only.
        mode='context_aware':  weighted combination of:
            - current sentence similarity
            - concatenated (current + prev sentence) similarity
            - pose feature similarity to prev_pose
        """
        if not self._built:
            raise RuntimeError("Call build_index() first.")
        if not self._index:
            return RetrievalResult(None, None, 0.0, None,
                                   "Dataset is empty — no samples indexed.", mode)

        candidates = self._index if not require_pose else [e for e in self._index if e["has_pose"]]
        if not candidates:
            return RetrievalResult(None, None, 0.0, None,
                                   "No pose files available. See DATASET_SETUP in README.", mode)

        model = self._get_model()

        # ── Text embedding for current sentence ────────────────────────────────
        cur_emb = model.encode([current_sentence], convert_to_numpy=True)[0]
        cur_emb = cur_emb / (np.linalg.norm(cur_emb) + 1e-9)

        # ── Context embedding ──────────────────────────────────────────────────
        ctx_emb = None
        if mode == "context_aware" and prev_sentence:
            combined = f"{prev_sentence} [SEP] {current_sentence}"
            ctx_emb = model.encode([combined], convert_to_numpy=True)[0]
            ctx_emb = ctx_emb / (np.linalg.norm(ctx_emb) + 1e-9)

        # ── Previous pose features ─────────────────────────────────────────────
        prev_pose_feat = None
        if mode == "context_aware" and prev_pose is not None:
            prev_pose_feat = summarize_pose(prev_pose)
            prev_pose_feat = prev_pose_feat / (np.linalg.norm(prev_pose_feat) + 1e-9)

        # ── Score each candidate ───────────────────────────────────────────────
        best_score = -np.inf
        best_entry = None

        for entry in candidates:
            idx_emb = entry["text_emb"]   # already normalised

            if mode == "text_only" or ctx_emb is None:
                score = float(np.dot(cur_emb, idx_emb))
            else:
                # Combined text score
                txt_score = (
                    self.TEXT_WEIGHT * float(np.dot(cur_emb, idx_emb)) +
                    (1 - self.TEXT_WEIGHT) * float(np.dot(ctx_emb, idx_emb))
                )
                # Pose similarity
                if prev_pose_feat is not None and entry["pose_feat"] is not None:
                    pf = entry["pose_feat"]
                    pf = pf / (np.linalg.norm(pf) + 1e-9)
                    pose_score = float(np.dot(prev_pose_feat, pf))
                    score = (1 - self.POSE_WEIGHT) * txt_score + self.POSE_WEIGHT * pose_score
                else:
                    score = txt_score

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < self.MATCH_THRESHOLD:
            return RetrievalResult(
                matched_sample_id=None,
                matched_sentence=None,
                score=float(best_score),
                pose=None,
                no_match_reason=(
                    f"Best match score {best_score:.2f} below threshold {self.MATCH_THRESHOLD:.2f}. "
                    "No reliable ASL pose available for this sentence."
                ),
                mode=mode,
            )

        return RetrievalResult(
            matched_sample_id=best_entry["sample_id"],
            matched_sentence=best_entry["sentence"],
            score=float(best_score),
            pose=best_entry["pose"],
            no_match_reason=None,
            mode=mode,
        )

    # ── Save / Load index ──────────────────────────────────────────────────────

    def save(self, path: str = "models/retrieval_index.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Don't save the SBERT model itself, only the index
        data = {
            "model_name": self.model_name,
            "is_synthetic": self.is_synthetic,
            "index": [
                {k: v for k, v in e.items() if k != "pose"}  # save without full pose arrays
                for e in self._index
            ],
            "index_with_pose": self._index,  # keep poses in index for inference
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Retrieval] Index saved to {path}")

    @classmethod
    def load(cls, path: str = "models/retrieval_index.pkl") -> "SignLingoRetrieval":
        with open(path, "rb") as f:
            data = pickle.load(f)
        engine = cls(model_name=data["model_name"])
        engine._index = data.get("index_with_pose", data.get("index", []))
        engine.is_synthetic = bool(data.get("is_synthetic", False))
        engine._built = True
        print(f"[Retrieval] Index loaded from {path}: {len(engine._index)} entries.")
        return engine

    @property
    def num_samples(self) -> int:
        return len(self._index)


# ── Transcript parser ─────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    index: int
    start_seconds: float
    end_seconds: float
    text: str


def parse_srt(content: str) -> List[TranscriptSegment]:
    """Parse SRT subtitle format."""
    segments = []
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
            times = lines[1].strip()
            start_s, end_s = [_srt_time(t) for t in times.split(" --> ")]
            text = " ".join(lines[2:]).strip()
            segments.append(TranscriptSegment(idx, start_s, end_s, text))
        except Exception:
            continue
    return segments


def parse_vtt(content: str) -> List[TranscriptSegment]:
    """Parse WebVTT subtitle format."""
    segments = []
    # Strip header
    lines = content.strip().splitlines()
    idx = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if " --> " in line:
            try:
                start_s, end_s = [_vtt_time(t) for t in line.split(" --> ")]
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip() != "":
                    text_lines.append(lines[i].strip())
                    i += 1
                text = " ".join(text_lines)
                idx += 1
                segments.append(TranscriptSegment(idx, start_s, end_s, text))
            except Exception:
                i += 1
        else:
            i += 1
    return segments


def _srt_time(t: str) -> float:
    t = t.strip().replace(",", ".")
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _vtt_time(t: str) -> float:
    t = t.strip()
    parts = t.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


if __name__ == "__main__":
    from ml.dataset import make_synthetic_dataset
    syn = make_synthetic_dataset(10)
    engine = SignLingoRetrieval()
    engine.build_index(syn)
    result = engine.retrieve("Hello, how are you?", mode="text_only")
    print(f"text_only  → {result.matched_sentence} (score={result.score:.3f})")
    result2 = engine.retrieve(
        "I am learning sign language.",
        mode="context_aware",
        prev_sentence="Hello, how are you?",
        prev_pose=syn[0].pose,
    )
    print(f"context_aware → {result2.matched_sentence} (score={result2.score:.3f})")
