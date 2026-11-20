"""Model status router."""

from fastapi import APIRouter

from backend.schemas import ModelStatusResponse
from backend.services.model_service import get_model_service

router = APIRouter(prefix="/model", tags=["Model"])


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status() -> ModelStatusResponse:
    """Return model checkpoint status, active weights path, and compute device readiness."""
    status_dict = get_model_service().get_status()
    return ModelStatusResponse(**status_dict)
