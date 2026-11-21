"""
SignLingo AI Mini — trained dual-encoder ranker.

A small text tower and pose tower are trained with a symmetric InfoNCE
(contrastive) objective on sentence-to-clip pairs. The text tower consumes
frozen all-MiniLM-L6-v2 embeddings (384-d); the pose tower consumes pooled
keypoint features (548-d). Both project into a shared 128-d space.

Training is real gradient descent (Adam). The fixed weighted cosine ranker
remains available as the baseline for comparison.

Checkpoint layout (models/ranker.pt):
    state_dict, config, model_name, seed, epochs, loss_log,
    train_sample_ids, val_sample_ids
"""

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is installed in this project
    HAS_TORCH = False

POSE_FEAT_DIM = 548  # temporal mean + std over 274 coords
TEXT_FEAT_DIM = 384  # all-MiniLM-L6-v2


def pose_features(pose: Optional[np.ndarray]) -> np.ndarray:
    """Compress a [T, 274] pose to temporal mean + std (548-d)."""
    if pose is None or len(pose) == 0:
        return np.zeros(POSE_FEAT_DIM, dtype=np.float32)
    mean = pose.mean(axis=0)
    std = pose.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLPTower(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class DualEncoder(nn.Module):
    def __init__(self, text_in: int = TEXT_FEAT_DIM, pose_in: int = POSE_FEAT_DIM,
                 hidden: int = 256, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.text_tower = MLPTower(text_in, hidden, out_dim, dropout)
        self.pose_tower = MLPTower(pose_in, hidden, out_dim, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.text_tower(x)

    def encode_pose(self, x: torch.Tensor) -> torch.Tensor:
        return self.pose_tower(x)


def info_nce(text_z: torch.Tensor, pose_z: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric in-batch InfoNCE."""
    logits = logit_scale.exp() * text_z @ pose_z.t()
    labels = torch.arange(text_z.shape[0], device=text_z.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train_on_features(
    text_feats: np.ndarray,
    pose_feats: np.ndarray,
    out_path: str,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    val_frac: float = 0.2,
    model_name: str = "all-MiniLM-L6-v2",
    config: Optional[Dict] = None,
    log_path: Optional[str] = None,
    val_indices: Optional[Sequence[int]] = None,
) -> Dict:
    """Train the dual encoder from precomputed features. Returns metadata dict."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for the trained ranker.")
    text_feats = np.asarray(text_feats, dtype=np.float32)
    pose_feats = np.asarray(pose_feats, dtype=np.float32)
    if text_feats.shape[0] != pose_feats.shape[0]:
        raise ValueError("text_feats and pose_feats must have the same number of rows.")
    n = text_feats.shape[0]
    if n < 8:
        raise ValueError("Need at least 8 samples to train the ranker.")

    set_seed(seed)
    rng = np.random.default_rng(seed)
    if val_indices is None:
        order = rng.permutation(n)
        n_val = max(1, int(round(n * val_frac)))
        val_idx = np.sort(order[:n_val])
        train_idx = np.sort(order[n_val:])
    else:
        val_idx = np.array(sorted(val_indices), dtype=np.int64)
        train_idx = np.array([i for i in range(n) if i not in set(val_idx.tolist())], dtype=np.int64)

    config = dict(config or {})
    defaults = {"text_in": text_feats.shape[1], "pose_in": pose_feats.shape[1],
                "hidden": 256, "out_dim": 128, "dropout": 0.1}
    defaults.update(config)
    model = DualEncoder(**defaults)

    text_t = torch.from_numpy(text_feats)
    pose_t = torch.from_numpy(pose_feats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_log: List[Dict] = []
    best_val = float("inf")
    best_state = None

    def batch_loss(idx: np.ndarray) -> torch.Tensor:
        t = text_t[idx]
        p = pose_t[idx]
        return info_nce(model.encode_text(t), model.encode_pose(p), model.logit_scale)

    model.train()
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(train_idx)
        total, count = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            loss = batch_loss(idx)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(idx)
            count += len(idx)
        train_loss = total / max(count, 1)

        model.eval()
        with torch.no_grad():
            if len(val_idx) >= 2:
                val_loss = float(batch_loss(val_idx).item())
            else:
                val_loss = float("nan")
        model.train()
        loss_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[ranker] epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss == val_loss and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": defaults,
        "model_name": model_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "loss_log": loss_log,
        "best_val_loss": best_val,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    print(f"[ranker] Saved checkpoint to {out_path}")

    log_path = log_path or str(Path(out_path).with_name("ranker_train_log.csv"))
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(loss_log)
    print(f"[ranker] Loss log: {log_path}")
    return {k: v for k, v in checkpoint.items() if k != "state_dict"}


def _sbert():
    from sentence_transformers import SentenceTransformer
    from ml.retrieval import _MODEL_CACHE

    name = "all-MiniLM-L6-v2"
    if name not in _MODEL_CACHE:
        print(f"[ranker] Loading sentence-transformer: {name}")
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def train_ranker(
    samples: Sequence,
    out_path: str = "models/ranker.pt",
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    val_frac: float = 0.2,
) -> Dict:
    """Encode real sentences + poses, then train the dual encoder."""
    samples = [s for s in samples if s.pose is not None]
    if len(samples) < 8:
        raise ValueError("Need at least 8 pose samples to train the ranker.")
    model = _sbert()
    print(f"[ranker] Encoding {len(samples)} captions with {model}…")
    text_feats = model.encode(
        [s.sentence for s in samples], convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    pose_feats = np.stack([pose_features(s.pose) for s in samples]).astype(np.float32)
    meta = train_on_features(
        text_feats, pose_feats, out_path=out_path, epochs=epochs,
        batch_size=batch_size, lr=lr, seed=seed, val_frac=val_frac,
        model_name="all-MiniLM-L6-v2",
    )
    meta["sample_ids"] = [s.sample_id for s in samples]
    return meta


@dataclass
class TrainedRetriever:
    """Frozen trained towers for encoding queries and clips."""
    model: "DualEncoder"
    checkpoint: Dict

    @classmethod
    def load(cls, path: str) -> "TrainedRetriever":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to load the trained ranker.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = DualEncoder(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return cls(model=model, checkpoint=checkpoint)

    @torch.no_grad()
    def encode_text(self, feats: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(feats, dtype=np.float32))
        return self.model.encode_text(tensor).cpu().numpy()

    @torch.no_grad()
    def encode_pose(self, feats: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(feats, dtype=np.float32))
        return self.model.encode_pose(tensor).cpu().numpy()

    @property
    def val_indices(self) -> List[int]:
        return list(self.checkpoint.get("val_indices", []))
