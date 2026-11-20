"""Experiments registry and metrics evaluation router."""

from fastapi import APIRouter, Query

from backend.schemas import ExperimentsResponse, MetricsResponse
from backend.services.experiment_service import get_experiment_service

router = APIRouter(tags=["Experiments"])


@router.get(
    "/experiments",
    response_model=ExperimentsResponse,
    summary="List all research experiments and execution states",
)
async def list_experiments() -> ExperimentsResponse:
    """Return all defined experiments (E1-E9) with live statuses derived from filesystem artifacts."""
    exp_service = get_experiment_service()
    experiments = exp_service.get_experiments()
    return ExperimentsResponse(experiments=experiments)


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get evaluation metrics for an experiment",
)
async def get_metrics(
    experiment_id: str = Query(
        default="e1",
        description="Experiment identifier (e.g., e1, e2, E1)",
    ),
) -> MetricsResponse:
    """Return evaluation metrics for an experiment, or 'not_evaluated' if results are not available."""
    exp_service = get_experiment_service()
    return exp_service.get_metrics(experiment_id=experiment_id)
