"""Experiment registry and metrics service for ContextSign AI."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import get_settings
from backend.schemas import ExperimentStatus, MetricsResponse

logger = logging.getLogger(__name__)

_RESEARCH_ROOT = Path(__file__).resolve().parents[2]


class ExperimentService:
    """Service to track experiment definitions, execution states, and evaluation metrics."""

    EXPERIMENT_DEFINITIONS = [
        {
            "id": "E1",
            "name": "Text-to-Pose Baseline",
            "default_status": "IMPLEMENTED",
            "description": "Text-only baseline model without context conditioning (context window = 0, CVAE = enabled).",
            "config_rel_path": "configs/e1_baseline.yaml",
            "checkpoint_name": "e1_best.pt",
        },
        {
            "id": "E2",
            "name": "Context-Aware ASL Generation (Primary)",
            "default_status": "IMPLEMENTED",
            "description": "Context-aware sign pose generation conditioned on 1 prior pose sequence using CrossAttentionFusion and CVAE.",
            "config_rel_path": "configs/e2_context_aware.yaml",
            "checkpoint_name": "e2_best.pt",
        },
        {
            "id": "E3",
            "name": "Multi-Sentence Context Window",
            "default_status": "IMPLEMENTED",
            "description": "Evaluates multi-utterance continuity by expanding context window to 2-3 prior sentences with CrossAttentionFusion.",
            "config_rel_path": "configs/e3_multisentence.yaml",
            "checkpoint_name": "e3_best.pt",
        },
        {
            "id": "E4",
            "name": "Text-Only Context Baseline",
            "default_status": "IMPLEMENTED",
            "description": "Conditioned on prior sentence English text embeddings instead of pose data to test multimodal vs text context.",
            "config_rel_path": "configs/e4_text_context.yaml",
            "checkpoint_name": "e4_best.pt",
        },
        {
            "id": "E5",
            "name": "Zero-Context Probe",
            "default_status": "IMPLEMENTED",
            "description": "Diagnostic probe evaluating E2 checkpoint under zeroed context features to quantify context reliance.",
            "config_rel_path": "configs/e5_zero_context.yaml",
            "checkpoint_name": "e2_best.pt",
        },
        {
            "id": "E6",
            "name": "Shuffled-Context Probe",
            "default_status": "IMPLEMENTED",
            "description": "Diagnostic probe evaluating E2 checkpoint with mismatched/shuffled prior poses to test semantic binding.",
            "config_rel_path": "configs/e6_shuffled_context.yaml",
            "checkpoint_name": "e2_best.pt",
        },
        {
            "id": "E7",
            "name": "Concatenation Fusion Ablation",
            "default_status": "IMPLEMENTED",
            "description": "Ablation study substituting CrossAttentionFusion with Concatenation and MLP projection.",
            "config_rel_path": "configs/e7_concat_fusion.yaml",
            "checkpoint_name": "e7_best.pt",
        },
        {
            "id": "E8",
            "name": "Deterministic Autoencoder Ablation",
            "default_status": "IMPLEMENTED",
            "description": "Ablation study removing CVAE stochastic latent space in favor of deterministic latent representation.",
            "config_rel_path": "configs/e8_deterministic.yaml",
            "checkpoint_name": "e8_best.pt",
        },
        {
            "id": "E9",
            "name": "Multimodal Context with Audio",
            "default_status": "PENDING",
            "description": "Proposed extension conditioning pose generation on both acoustic speech features and prior poses.",
            "config_rel_path": "configs/e9_multimodal_audio.yaml",
            "checkpoint_name": "e9_best.pt",
        },
    ]

    def __init__(self):
        settings = get_settings()
        self.configs_dir = _RESEARCH_ROOT / "configs"
        self.results_dir = _RESEARCH_ROOT / "results"
        self.checkpoints_dir = (
            Path(settings.CHECKPOINT_DIR)
            if Path(settings.CHECKPOINT_DIR).is_absolute()
            else _RESEARCH_ROOT / settings.CHECKPOINT_DIR
        )

    def _resolve_status(self, exp_meta: dict) -> str:
        """Determine actual experiment status based on filesystem artifacts."""
        exp_id = exp_meta["id"]
        exp_id_lower = exp_id.lower()

        # Check for results file first -> EVALUATED
        eval_candidates = [
            self.results_dir / f"{exp_id_lower}_eval.json",
            self.results_dir / f"{exp_id_lower}_metrics.json",
            self.results_dir / f"{exp_id_lower}.json",
            self.results_dir / f"{exp_id}_eval.json",
            self.results_dir / f"{exp_id}_metrics.json",
            self.results_dir / f"{exp_id}.json",
        ]
        for c in eval_candidates:
            if c.is_file():
                return "EVALUATED"

        # Check for trained checkpoint -> TRAINED
        ckpt_name = exp_meta.get("checkpoint_name")
        if ckpt_name:
            ckpt_path = self.checkpoints_dir / ckpt_name
            if ckpt_path.is_file():
                return "TRAINED"

        return exp_meta["default_status"]

    def get_experiments(self) -> List[ExperimentStatus]:
        """Return full list of research experiments with real-time resolved status."""
        results = []
        for exp in self.EXPERIMENT_DEFINITIONS:
            cfg_path = _RESEARCH_ROOT / exp["config_rel_path"]
            resolved_cfg = str(cfg_path) if cfg_path.is_file() else None

            results.append(
                ExperimentStatus(
                    id=exp["id"],
                    name=exp["name"],
                    status=self._resolve_status(exp),
                    description=exp["description"],
                    config_path=resolved_cfg,
                )
            )
        return results

    def get_metrics(self, experiment_id: str) -> MetricsResponse:
        """Load real evaluation metrics for experiment if results exist, otherwise return not_evaluated."""
        exp_id_norm = experiment_id.strip().upper()
        exp_id_lower = exp_id_norm.lower()

        now_str = datetime.now(timezone.utc).isoformat()

        candidate_files = [
            self.results_dir / f"{exp_id_lower}_eval.json",
            self.results_dir / f"{exp_id_lower}_metrics.json",
            self.results_dir / f"{exp_id_lower}.json",
            self.results_dir / f"{exp_id_norm}_eval.json",
            self.results_dir / f"{exp_id_norm}_metrics.json",
            self.results_dir / f"{exp_id_norm}.json",
        ]

        for file_path in candidate_files:
            if file_path.is_file():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Determine data type from content or filename
                    data_type = "real_how2sign"
                    if "synthetic" in str(data).lower() or "smoke" in str(data).lower():
                        data_type = "synthetic_smoke_test"

                    metrics_dict = data.get("metrics", data) if isinstance(data, dict) else {"raw": data}
                    recorded_time = (
                        data.get("timestamp", now_str)
                        if isinstance(data, dict)
                        else now_str
                    )

                    return MetricsResponse(
                        experiment_id=exp_id_norm,
                        metrics=metrics_dict,
                        data_type=data_type,
                        timestamp=str(recorded_time),
                    )
                except Exception as e:
                    logger.error("Failed to parse metrics file %s: %s", file_path, e)

        # No metrics file found
        return MetricsResponse(
            experiment_id=exp_id_norm,
            metrics={},
            data_type="not_evaluated",
            timestamp=now_str,
        )


# Global singleton instance
_experiment_service: Optional[ExperimentService] = None


def get_experiment_service() -> ExperimentService:
    """Return singleton instance of ExperimentService."""
    global _experiment_service
    if _experiment_service is None:
        _experiment_service = ExperimentService()
    return _experiment_service
