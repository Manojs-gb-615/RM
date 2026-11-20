"""Dataset samples and metadata router."""

from fastapi import APIRouter, HTTPException, Query, status

from backend.schemas import DatasetSample, DatasetSamplesResponse, ErrorResponse
from backend.services.dataset_service import get_dataset_service

router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.get(
    "/samples",
    response_model=DatasetSamplesResponse,
    summary="List paginated How2Sign dataset samples",
)
async def list_dataset_samples(
    split: str = Query(default="train", description="Dataset split (train, val, test)"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of samples to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
) -> DatasetSamplesResponse:
    """Return paginated list of real How2Sign dataset samples including context pairs."""
    dataset_service = get_dataset_service()
    samples = dataset_service.get_samples(split=split, limit=limit, offset=offset)
    total = dataset_service.get_total(split=split)

    return DatasetSamplesResponse(
        samples=samples,
        total=total,
        split=split,
    )


@router.get(
    "/samples/{sample_id:path}",
    response_model=DatasetSample,
    responses={
        404: {"model": ErrorResponse, "description": "Sample not found"},
    },
    summary="Get single dataset sample by ID",
)
async def get_dataset_sample(sample_id: str) -> DatasetSample:
    """Retrieve metadata and context pair for a specific sample ID."""
    dataset_service = get_dataset_service()
    sample = dataset_service.get_sample(sample_id)
    if not sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset sample '{sample_id}' not found in loaded splits.",
        )
    return sample
