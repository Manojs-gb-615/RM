"""
SignLingo AI Mini — Inference
Loads a saved retrieval index and runs pose retrieval on demand.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from ml.retrieval import SignLingoRetrieval, RetrievalResult, NO_POSE_RESULT
from ml.dataset import SignLingoDataset, PoseSample


class SignLingoInference:
    """
    Wraps SignLingoRetrieval for use by the FastAPI backend.
    Handles index loading, missing files, and serialization.
    """

    INDEX_PATH = "models/retrieval_index.pkl"

    def __init__(self):
        self._engine: Optional[SignLingoRetrieval] = None
        self._dataset: Optional[SignLingoDataset] = None
        self._ready = False

    def setup(self, data_dir: str = "data", pose_dir: Optional[str] = None) -> bool:
        """
        Load dataset + retrieval index. Returns True if ready.
        Called once at backend startup.
        """
        # Load dataset (for sample browsing)
        self._dataset = SignLingoDataset(data_dir=data_dir, pose_dir=pose_dir).load()

        # Load retrieval index if it exists
        if Path(self.INDEX_PATH).exists():
            try:
                self._engine = SignLingoRetrieval.load(self.INDEX_PATH)
                self._ready = True
                print(f"[Inference] Retrieval engine ready. {self._engine.num_samples} samples indexed.")
            except Exception as e:
                print(f"[Inference] Failed to load index: {e}")
                self._ready = False
        else:
            # Try to build index from dataset samples with poses
            samples_with_pose = self._dataset.samples_with_pose()
            if samples_with_pose:
                print(f"[Inference] No saved index — building from {len(samples_with_pose)} pose samples.")
                self._engine = SignLingoRetrieval()
                self._engine.build_index(samples_with_pose)
                self._engine.save(self.INDEX_PATH)
                self._ready = True
            else:
                print("[Inference] No pose data found. Engine not ready. "
                      "Download How2Sign poses or run training notebook first.")
                self._ready = False

        return self._ready

    @property
    def is_ready(self) -> bool:
        return self._ready

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "index_path": self.INDEX_PATH,
            "index_exists": Path(self.INDEX_PATH).exists(),
            "num_indexed": self._engine.num_samples if self._engine else 0,
            "dataset_samples": len(self._dataset.samples) if self._dataset else 0,
            "dataset_with_pose": len(self._dataset.samples_with_pose()) if self._dataset else 0,
            "message": (
                "Retrieval engine ready." if self._ready
                else "No pose data available. See README for setup instructions."
            ),
        }

    def retrieve_for_sentence(
        self,
        current_sentence: str,
        mode: str = "text_only",
        prev_sentence: Optional[str] = None,
        prev_pose_list: Optional[list] = None,  # serialized as list of lists
    ) -> Dict[str, Any]:
        """
        Run retrieval and return a JSON-serializable dict.
        """
        if not self._ready or self._engine is None:
            return {
                "matched_sample_id": None,
                "matched_sentence": None,
                "score": 0.0,
                "pose": None,
                "no_match_reason": "Retrieval engine not ready. No pose data available.",
                "mode": mode,
                "ready": False,
            }

        prev_pose = None
        if prev_pose_list is not None:
            try:
                prev_pose = np.array(prev_pose_list, dtype=np.float32)
            except Exception:
                prev_pose = None

        result: RetrievalResult = self._engine.retrieve(
            current_sentence=current_sentence,
            mode=mode,
            prev_sentence=prev_sentence,
            prev_pose=prev_pose,
            require_pose=True,
        )

        return {
            "matched_sample_id": result.matched_sample_id,
            "matched_sentence": result.matched_sentence,
            "score": result.score,
            "pose": result.pose.tolist() if result.pose is not None else None,
            "no_match_reason": result.no_match_reason,
            "mode": result.mode,
            "ready": True,
        }

    def get_dataset_samples(self) -> list:
        if self._dataset is None:
            return []
        return self._dataset.to_dict_list()

    def get_sample_pose(self, sample_id: str) -> Optional[list]:
        if self._dataset is None:
            return None
        s = self._dataset.get_sample(sample_id)
        if s is None or s.pose is None:
            return None
        return s.pose.tolist()
