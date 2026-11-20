"""Dataset and data loading utilities for How2Sign and Synthetic ASL data."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class DummyTokenizer:
    """Fallback tokenizer if transformers is unavailable or offline."""

    def __init__(self, vocab_size: int = 30522, max_len: int = 64):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: str = "max_length",
        max_length: int = 64,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text

        batch_input_ids = []
        batch_attention_mask = []

        for item in text_list:
            words = item.split()
            # Deterministic pseudo-tokenization
            token_ids = [self.cls_token_id]
            for w in words:
                token_ids.append((abs(hash(w)) % (self.vocab_size - 1000)) + 1000)
            token_ids.append(self.sep_token_id)

            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[: max_length - 1] + [self.sep_token_id]

            mask = [1] * len(token_ids)
            if padding == "max_length" and len(token_ids) < max_length:
                pad_len = max_length - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_len)
                mask.extend([0] * pad_len)

            batch_input_ids.append(token_ids)
            batch_attention_mask.append(mask)

        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)

        if isinstance(text, str):
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Load Hugging Face tokenizer with automatic fallback."""
    if HF_AVAILABLE:
        try:
            return AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        except Exception:
            try:
                return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            except Exception:
                pass
    return DummyTokenizer()


class SyntheticPoseDataset(Dataset):
    """Generates synthetic text and pose data for smoke testing, debugging, and verification."""

    SAMPLE_SENTENCES = [
        "Hello, welcome to American Sign Language tutorial.",
        "Today we will learn how to prepare a delicious dinner.",
        "Please turn right at the traffic lights.",
        "Sign language uses facial expressions and hand shapes.",
        "Good morning everyone, how are you doing today?",
        "First take the flour and pour it into the large bowl.",
        "Can you repeat what you just said more slowly?",
        "Thank you very much for helping us with this project.",
        "The weather is very sunny and warm outside.",
        "We are evaluating context-aware sign language generation.",
    ]

    def __init__(
        self,
        num_samples: int = 100,
        pose_dim: int = 274,
        max_pose_frames: int = 128,
        max_text_tokens: int = 64,
        context_window: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.pose_dim = pose_dim
        self.max_pose_frames = max_pose_frames
        self.max_text_tokens = max_text_tokens
        self.context_window = context_window
        self.rng = np.random.RandomState(seed)

        self.tokenizer = get_tokenizer("distilbert-base-uncased")

        # Pre-generate synthetic sample descriptors
        self.samples = []
        for i in range(num_samples):
            # Target frame count between 20 and max_pose_frames
            n_frames = self.rng.randint(20, max_pose_frames + 1)
            sentence = self.SAMPLE_SENTENCES[i % len(self.SAMPLE_SENTENCES)]
            self.samples.append(
                {
                    "sample_id": f"synthetic_{i:04d}",
                    "sentence": sentence,
                    "n_frames": n_frames,
                }
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self.samples[idx]
        sentence = info["sentence"]
        n_frames = info["n_frames"]

        # Tokenize text
        tokens = self.tokenizer(
            sentence,
            padding="max_length",
            max_length=self.max_text_tokens,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        if input_ids.ndim == 2:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)

        # Generate smooth synthetic pose using sinusoidal curves + noise
        t = np.linspace(0, 2 * np.pi, n_frames)[:, None]
        freq = self.rng.uniform(0.5, 2.0, size=(1, self.pose_dim))
        phase = self.rng.uniform(0, np.pi, size=(1, self.pose_dim))
        target_pose_np = np.sin(t * freq + phase) * 0.5 + self.rng.normal(0, 0.05, size=(n_frames, self.pose_dim))

        target_pose = torch.tensor(target_pose_np, dtype=torch.float32)
        target_mask = torch.ones(n_frames, dtype=torch.float32)

        # Generate context poses: [context_window, T, pose_dim]
        if self.context_window > 0:
            ctx_list = []
            for _ in range(self.context_window):
                ctx_frames = self.rng.randint(20, self.max_pose_frames + 1)
                t_ctx = np.linspace(0, 2 * np.pi, ctx_frames)[:, None]
                ctx_pose_np = np.sin(t_ctx * freq) * 0.4 + self.rng.normal(0, 0.05, size=(ctx_frames, self.pose_dim))
                # Pad to max_pose_frames for tensor stacking
                padded = np.zeros((self.max_pose_frames, self.pose_dim), dtype=np.float32)
                padded[:ctx_frames] = ctx_pose_np
                ctx_list.append(padded)
            context_poses = torch.tensor(np.stack(ctx_list, axis=0), dtype=torch.float32)
        else:
            context_poses = torch.zeros((0, self.max_pose_frames, self.pose_dim), dtype=torch.float32)

        return {
            "sample_id": info["sample_id"],
            "sentence": sentence,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_pose": target_pose,
            "target_mask": target_mask,
            "context_poses": context_poses,
        }


class How2SignPoseDataset(Dataset):
    """Dataset for real How2Sign pose data loaded from manifests and numpy files."""

    def __init__(
        self,
        manifest_path: str,
        pose_root: str,
        context_window: int = 1,
        norm_stats_path: Optional[str] = None,
        max_pose_frames: int = 128,
        max_text_tokens: int = 64,
        pose_dim: int = 274,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.pose_root = Path(pose_root)
        self.context_window = context_window
        self.max_pose_frames = max_pose_frames
        self.max_text_tokens = max_text_tokens
        self.pose_dim = pose_dim

        # Check that pose_root exists and contains pose files
        if not self.pose_root.exists():
            raise FileNotFoundError(
                f"Pose root directory '{self.pose_root}' does not exist. "
                "How2Sign pose data (.npy/.npz) must be downloaded first before running on real data."
            )

        pose_files = list(self.pose_root.glob("*.npy")) + list(self.pose_root.glob("*.npz"))
        if len(pose_files) == 0:
            raise FileNotFoundError(
                f"Pose root directory '{self.pose_root}' contains no .npy or .npz files. "
                "How2Sign pose files must be downloaded and extracted before training/evaluating."
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file '{self.manifest_path}' does not exist.")

        import pandas as pd
        self.df = pd.read_csv(self.manifest_path)
        self.tokenizer = get_tokenizer("distilbert-base-uncased")

        # Load normalization statistics if provided
        self.mean = None
        self.std = None
        if norm_stats_path and os.path.exists(norm_stats_path):
            stats = np.load(norm_stats_path)
            self.mean = stats["mean"]
            self.std = stats["std"]

    def __len__(self) -> int:
        return len(self.df)

    def _load_pose(self, sample_id: str) -> Optional[np.ndarray]:
        npy_path = self.pose_root / f"{sample_id}.npy"
        npz_path = self.pose_root / f"{sample_id}.npz"
        if npy_path.exists():
            return np.load(npy_path)
        elif npz_path.exists():
            data = np.load(npz_path)
            if "pose" in data:
                return data["pose"]
            return data[list(data.keys())[0]]
        return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample_id = str(row["sample_id"])
        sentence = str(row.get("sentence", ""))

        # Load target pose
        pose_data = self._load_pose(sample_id)
        if pose_data is None:
            raise FileNotFoundError(f"Pose file for sample '{sample_id}' not found in {self.pose_root}")

        # Ensure 2D [T, pose_dim]
        if pose_data.ndim == 3:
            pose_data = pose_data.reshape(pose_data.shape[0], -1)

        if pose_data.shape[1] != self.pose_dim:
            # Reshape or pad/truncate to expected pose_dim
            if pose_data.shape[1] > self.pose_dim:
                pose_data = pose_data[:, : self.pose_dim]
            else:
                pad_width = self.pose_dim - pose_data.shape[1]
                pose_data = np.pad(pose_data, ((0, 0), (0, pad_width)))

        # Normalize if stats available
        if self.mean is not None and self.std is not None:
            pose_data = (pose_data - self.mean) / (self.std + 1e-8)

        # Truncate if exceeds max_pose_frames
        n_frames = min(len(pose_data), self.max_pose_frames)
        target_pose_np = pose_data[:n_frames]

        target_pose = torch.tensor(target_pose_np, dtype=torch.float32)
        target_mask = torch.ones(n_frames, dtype=torch.float32)

        # Tokenize text
        tokens = self.tokenizer(
            sentence,
            padding="max_length",
            max_length=self.max_text_tokens,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        if input_ids.ndim == 2:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)

        # Load context poses
        if self.context_window > 0:
            ctx_list = []
            for w in range(1, self.context_window + 1):
                prev_col = f"prev_sentence_id_{w}"
                prev_id = str(row.get(prev_col, "")) if prev_col in row else ""
                prev_pose = self._load_pose(prev_id) if prev_id and prev_id != "nan" else None

                padded = np.zeros((self.max_pose_frames, self.pose_dim), dtype=np.float32)
                if prev_pose is not None:
                    if prev_pose.ndim == 3:
                        prev_pose = prev_pose.reshape(prev_pose.shape[0], -1)
                    if self.mean is not None and self.std is not None:
                        prev_pose = (prev_pose - self.mean) / (self.std + 1e-8)
                    p_len = min(len(prev_pose), self.max_pose_frames)
                    padded[:p_len] = prev_pose[:p_len, : self.pose_dim]
                ctx_list.append(padded)
            context_poses = torch.tensor(np.stack(ctx_list, axis=0), dtype=torch.float32)
        else:
            context_poses = torch.zeros((0, self.max_pose_frames, self.pose_dim), dtype=torch.float32)

        return {
            "sample_id": sample_id,
            "sentence": sentence,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_pose": target_pose,
            "target_mask": target_mask,
            "context_poses": context_poses,
        }


def collate_fn(batch: List[Dict[str, Union[torch.Tensor, str]]]) -> Dict[str, torch.Tensor]:
    """Collates and pads variable-length sequences into a uniform batch."""
    batch_size = len(batch)
    sample_ids = [item["sample_id"] for item in batch]
    sentences = [item["sentence"] for item in batch]

    # Text token tensors
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)

    # Pose sequence lengths
    pose_lengths = [item["target_pose"].shape[0] for item in batch]
    max_len = max(pose_lengths)
    pose_dim = batch[0]["target_pose"].shape[1]

    # Pad target_pose and target_mask
    padded_poses = torch.zeros((batch_size, max_len, pose_dim), dtype=torch.float32)
    padded_masks = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for i, item in enumerate(batch):
        t_len = item["target_pose"].shape[0]
        padded_poses[i, :t_len] = item["target_pose"]
        padded_masks[i, :t_len] = item["target_mask"]

    # Context poses
    # Check context_window
    first_ctx = batch[0]["context_poses"]
    context_window = first_ctx.shape[0]
    if context_window > 0:
        # Context poses are [context_window, T_ctx, pose_dim]
        max_ctx_len = max(item["context_poses"].shape[1] for item in batch)
        padded_ctx = torch.zeros((batch_size, context_window, max_ctx_len, pose_dim), dtype=torch.float32)
        for i, item in enumerate(batch):
            c_poses = item["context_poses"]
            padded_ctx[i, :, : c_poses.shape[1]] = c_poses
        context_poses = padded_ctx
    else:
        context_poses = torch.zeros((batch_size, 0, max_len, pose_dim), dtype=torch.float32)

    return {
        "sample_id": sample_ids,
        "sentence": sentences,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_pose": padded_poses,
        "target_mask": padded_masks,
        "context_poses": context_poses,
    }


def compute_normalization_stats(
    dataset: Dataset, save_path: str = "data/norm_stats.npz"
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes global mean and std across all valid frames in dataset and saves to npz."""
    all_frames = []
    for i in range(len(dataset)):
        item = dataset[i]
        pose = item["target_pose"].cpu().numpy()
        mask = item["target_mask"].cpu().numpy()
        valid_pose = pose[mask > 0.5]
        if len(valid_pose) > 0:
            all_frames.append(valid_pose)

    if len(all_frames) == 0:
        # Fallback to zeros/ones
        mean = np.zeros(274, dtype=np.float32)
        std = np.ones(274, dtype=np.float32)
    else:
        stacked = np.concatenate(all_frames, axis=0)
        mean = np.mean(stacked, axis=0).astype(np.float32)
        std = np.std(stacked, axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez_compressed(save_path, mean=mean, std=std)
    return mean, std
