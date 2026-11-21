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

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", trained=None):
        self.model_name = model_name
        self._sbert: Optional[SentenceTransformer] = None
        self._index: List[Dict] = []   # [{sample_id, sentence, text_emb, pose_feat}]
        self._built = False
        self.is_synthetic = False
        self._trained = trained
        self.trained_checkpoint: Optional[str] = None
        self._text_mat: Optional[np.ndarray] = None
        self._pose_feat_mat: Optional[np.ndarray] = None
        self._trained_text_mat: Optional[np.ndarray] = None
        self._trained_pose_mat: Optional[np.ndarray] = None
        self._has_pose: Optional[np.ndarray] = None

    @property
    def has_trained(self) -> bool:
        return self._trained is not None

    def _rebuild_matrices(self) -> None:
        n = len(self._index)
        if n == 0:
            self._text_mat = None
            self._pose_feat_mat = None
            self._trained_text_mat = None
            self._trained_pose_mat = None
            self._has_pose = None
            return
        self._text_mat = np.stack([e["text_emb"] for e in self._index]).astype(np.float32)
        pose_feats = []
        for e in self._index:
            pf = e.get("pose_feat")
            if pf is None:
                pose_feats.append(np.zeros(64, dtype=np.float32))
            else:
                norm = np.linalg.norm(pf) + 1e-9
                pose_feats.append((pf / norm).astype(np.float32))
        self._pose_feat_mat = np.stack(pose_feats)
        self._has_pose = np.array([bool(e.get("has_pose")) for e in self._index], dtype=bool)
        if self.has_trained and "trained_text_emb" in self._index[0]:
            self._trained_text_mat = np.stack(
                [e["trained_text_emb"] for e in self._index]
            ).astype(np.float32)
            self._trained_pose_mat = np.stack(
                [e["trained_pose_emb"] for e in self._index]
            ).astype(np.float32)
        else:
            self._trained_text_mat = None
            self._trained_pose_mat = None

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

    def build_index(self, samples: List[PoseSample], trained=None) -> "SignLingoRetrieval":
        """Encode all sample sentences and cache pose features.

        If a trained dual encoder is supplied, also cache its projected
        text/pose embeddings so retrieval uses the trained ranker.
        """
        if trained is not None:
            self._trained = trained
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

        from ml.ranker import pose_features

        trained_text = None
        trained_pose = None
        if self._trained is not None:
            print("[Retrieval] Encoding clips with trained dual encoder…")
            trained_text = self._trained.encode_text(embeddings)
            pose_matrix = np.stack([
                pose_features(s.pose) if s.pose is not None else np.zeros(548, dtype=np.float32)
                for s in samples
            ]).astype(np.float32)
            trained_pose = self._trained.encode_pose(pose_matrix)

        self._index = []
        for i, (s, emb) in enumerate(zip(samples, embeddings)):
            entry = {
                "sample_id": s.sample_id,
                "sentence": s.sentence,
                "text_emb": emb,
                "pose_feat": summarize_pose(s.pose) if s.pose is not None else None,
                "pose": s.pose,
                "has_pose": s.pose is not None,
            }
            if trained_text is not None and trained_pose is not None:
                entry["trained_text_emb"] = trained_text[i]
                entry["trained_pose_emb"] = trained_pose[i]
            self._index.append(entry)
        self._rebuild_matrices()
        self._built = True
        print(f"[Retrieval] Index built: {len(self._index)} entries, "
              f"{sum(1 for e in self._index if e['has_pose'])} with pose, "
              f"trained={'yes' if self.has_trained else 'no'}.")
        return self

    # ── Retrieve ───────────────────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        model = self._get_model()
        emb = model.encode([text], convert_to_numpy=True)[0]
        return emb / (np.linalg.norm(emb) + 1e-9)

    def score_all(
        self,
        current_sentence: str,
        mode: str = "text_only",
        prev_sentence: Optional[str] = None,
        prev_pose: Optional[np.ndarray] = None,
        require_pose: bool = True,
        include_current: bool = True,
    ) -> List[Tuple[float, Dict]]:
        """Score every indexed clip and return [(score, entry)] descending.

        include_current=False is used only by the context-only ablation in
        ml/evaluate.py: it hides the current sentence so retrieval must rely on
        context + previous pose.
        """
        if not self._built:
            raise RuntimeError("Call build_index() first.")
        if not self._index:
            return []
        if self._text_mat is None:
            self._rebuild_matrices()

        n = len(self._index)
        mask = self._has_pose.copy() if require_pose else np.ones(n, dtype=bool)
        if not mask.any():
            return []

        cur_emb = self._encode(current_sentence) if include_current else None
        ctx_emb = None
        if mode == "context_aware" and prev_sentence:
            combined = f"{prev_sentence} [SEP] {current_sentence}" if include_current else prev_sentence
            ctx_emb = self._encode(combined)
        prev_pose_feat = None
        if mode == "context_aware" and prev_pose is not None:
            pf = summarize_pose(prev_pose)
            prev_pose_feat = pf / (np.linalg.norm(pf) + 1e-9)

        if self.has_trained and self._trained_text_mat is not None:
            scores = self._score_trained(cur_emb, ctx_emb, prev_pose, mode, include_current)
        else:
            scores = self._score_baseline(cur_emb, ctx_emb, prev_pose_feat, mode, include_current)

        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)
        return [(float(scores[i]), self._index[i]) for i in order if np.isfinite(scores[i])]

    def _score_baseline(self, cur_emb, ctx_emb, prev_pose_feat, mode, include_current) -> np.ndarray:
        cur_scores = self._text_mat @ cur_emb if cur_emb is not None else np.zeros(len(self._index))
        if mode != "context_aware" or ctx_emb is None:
            return cur_scores.astype(np.float64)
        ctx_scores = self._text_mat @ ctx_emb
        if include_current:
            txt = self.TEXT_WEIGHT * cur_scores + (1 - self.TEXT_WEIGHT) * ctx_scores
        else:
            txt = ctx_scores
        scores = txt.astype(np.float64)
        if prev_pose_feat is not None:
            pose_scores = self._pose_feat_mat @ prev_pose_feat
            scores = (1 - self.POSE_WEIGHT) * scores + self.POSE_WEIGHT * pose_scores
        return scores

    def _score_trained(self, cur_emb, ctx_emb, prev_pose, mode, include_current) -> np.ndarray:
        from ml.ranker import pose_features

        text_mat = self._trained_text_mat
        cur_scores = np.zeros(len(self._index), dtype=np.float64)
        ctx_scores = np.zeros(len(self._index), dtype=np.float64)
        pose_scores = np.zeros(len(self._index), dtype=np.float64)
        have_cur = cur_emb is not None
        have_ctx = ctx_emb is not None
        have_pose = prev_pose is not None

        if have_cur:
            q = self._trained.encode_text(cur_emb[None])[0]
            cur_scores = text_mat @ q
        if have_ctx:
            q = self._trained.encode_text(ctx_emb[None])[0]
            ctx_scores = text_mat @ q
        if have_pose:
            q = self._trained.encode_pose(pose_features(prev_pose)[None])[0]
            pose_scores = self._trained_pose_mat @ q

        weights = {"text": 0.0, "ctx": 0.0, "pose": 0.0}
        if mode == "context_aware" and have_cur:
            weights["text"] = 0.6
        elif have_cur:
            weights["text"] = 1.0
        if mode == "context_aware" and have_ctx:
            weights["ctx"] = 0.2
        if mode == "context_aware" and have_pose:
            weights["pose"] = 0.2
        total = sum(weights.values())
        if total <= 0:
            weights["text"] = 1.0
            total = 1.0
        scores = (
            weights["text"] * cur_scores
            + weights["ctx"] * ctx_scores
            + weights["pose"] * pose_scores
        ) / total
        return scores.astype(np.float64)

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
        mode='text_only':      sentence similarity only.
        mode='context_aware':  current + previous sentence + previous pose.
        """
        if not self._built:
            raise RuntimeError("Call build_index() first.")
        if not self._index:
            return RetrievalResult(None, None, 0.0, None,
                                   "Dataset is empty — no samples indexed.", mode)

        candidates = [e for e in self._index if e["has_pose"]] if require_pose else self._index
        if not candidates:
            return RetrievalResult(None, None, 0.0, None,
                                   "No pose files available. See DATASET_SETUP in README.", mode)

        ranked = self.score_all(
            current_sentence=current_sentence, mode=mode,
            prev_sentence=prev_sentence, prev_pose=prev_pose,
            require_pose=require_pose,
        )
        if not ranked:
            return RetrievalResult(None, None, 0.0, None,
                                   "No pose files available. See DATASET_SETUP in README.", mode)

        best_score, best_entry = ranked[0]

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
        # Don't save the SBERT model itself, only the index entries.
        data = {
            "model_name": self.model_name,
            "is_synthetic": self.is_synthetic,
            "use_trained": self.has_trained,
            "trained_checkpoint": self.trained_checkpoint,
            "index_with_pose": self._index,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Retrieval] Index saved to {path} "
              f"({'trained' if self.has_trained else 'baseline'} ranker)")

    @classmethod
    def load(cls, path: str = "models/retrieval_index.pkl") -> "SignLingoRetrieval":
        with open(path, "rb") as f:
            data = pickle.load(f)
        engine = cls(model_name=data["model_name"])
        engine._index = data.get("index_with_pose", data.get("index", []))
        engine.is_synthetic = bool(data.get("is_synthetic", False))
        engine.trained_checkpoint = data.get("trained_checkpoint")
        if data.get("use_trained"):
            checkpoint = engine.trained_checkpoint or "models/ranker.pt"
            try:
                from ml.ranker import TrainedRetriever
                engine._trained = TrainedRetriever.load(checkpoint)
                print(f"[Retrieval] Loaded trained ranker: {checkpoint}")
            except Exception as exc:
                print(f"[Retrieval] WARNING: trained ranker unavailable ({exc}); "
                      "falling back to baseline cosine.")
                engine._trained = None
        engine._rebuild_matrices()
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
