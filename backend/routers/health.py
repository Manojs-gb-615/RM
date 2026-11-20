"""Health check endpoint."""

from datetime import datetime, timezone
from fastapi import APIRouter

from backend.schemas import HealthResponse
from backend.services.model_service import get_model_service

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Return system operational status, model checkpoint readiness, and CUDA GPU status."""
    cuda_available = False
    device_name = "cpu"

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = f"cuda:0 ({torch.cuda.get_device_name(0)})"
    except ImportError:
        pass

    model_service = get_model_service()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_ready=model_service.is_ready(),
        cuda_available=cuda_available,
        device=device_name,
    )
