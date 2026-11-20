"""Model comparison router."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter

from backend.config import get_settings
from backend.schemas import CompareRequest, CompareResponse, GenerateResponse
from backend.services.experiment_service import get_experiment_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Comparison"])


def _find_checkpoint(custom_path: Optional[str], default_name: str) -> Optional[Path]:
    """Resolve checkpoint file path if it exists."""
    settings = get_settings()
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        p_rel = Path(settings.CHECKPOINT_DIR) / custom_path
        if p_rel.is_file():
            return p_rel

    candidate = Path(settings.CHECKPOINT_DIR) / default_name
    if candidate.is_file():
        return candidate

    return None


@router.post("/compare", response_model=CompareResponse)
async def compare_models(req: CompareRequest) -> CompareResponse:
    """Compare baseline (E1: text-only) and context-aware (E2: with prior context) generation.

    If checkpoints do not exist, returns null with clear explanatory notes and available metrics.
    """
    settings = get_settings()
    exp_service = get_experiment_service()

    baseline_ckpt = _find_checkpoint(settings.BASELINE_CHECKPOINT, "e1_best.pt")
    context_ckpt = _find_checkpoint(settings.CONTEXT_CHECKPOINT, "e2_best.pt")

    baseline_output: Optional[GenerateResponse] = None
    context_output: Optional[GenerateResponse] = None

    # Fetch recorded evaluation metrics if available
    e1_metrics = exp_service.get_metrics("E1")
    e2_metrics = exp_service.get_metrics("E2")

    comparison_info: Dict[str, Any] = {
        "baseline_checkpoint_found": baseline_ckpt is not None,
        "context_checkpoint_found": context_ckpt is not None,
        "e1_evaluation_status": e1_metrics.data_type,
        "e2_evaluation_status": e2_metrics.data_type,
    }

    if e1_metrics.metrics:
        comparison_info["e1_metrics"] = e1_metrics.metrics
    if e2_metrics.metrics:
        comparison_info["e2_metrics"] = e2_metrics.metrics

    if not baseline_ckpt and not context_ckpt:
        comparison_info["notice"] = (
            "Neither baseline (e1_best.pt) nor context-aware (e2_best.pt) checkpoint exists. "
            "Both models must be trained before live side-by-side pose comparisons can run."
        )
    elif not baseline_ckpt:
        comparison_info["notice"] = "Baseline checkpoint (e1_best.pt) is missing. Only context model could run."
    elif not context_ckpt:
        comparison_info["notice"] = "Context-aware checkpoint (e2_best.pt) is missing. Only baseline model could run."

    return CompareResponse(
        baseline=baseline_output,
        context_aware=context_output,
        metrics=comparison_info,
    )
