"""Sign pose generation router."""

import asyncio
import logging
from typing import Union
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.schemas import ErrorResponse, GenerateRequest, GenerateResponse
from backend.services.model_service import ModelNotReadyError, get_model_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Generation"])

RESEARCH_WARNING = (
    "ContextSign AI is currently in active research and evaluation. "
    "Generated ASL pose sequences are experimental and require validation."
)


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model checkpoint not loaded or not ready"},
        504: {"model": ErrorResponse, "description": "Inference computation timed out"},
    },
)
async def generate_poses(req: GenerateRequest) -> Union[GenerateResponse, JSONResponse]:
    """Generate ASL keypoint pose sequences from input text and optional context poses."""
    model_service = get_model_service()

    if not model_service.is_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "ModelNotReady",
                "detail": (
                    "No trained model checkpoint is available for inference. "
                    "ContextSign AI requires a trained model checkpoint (.pt) before pose generation can occur."
                ),
                "suggestion": (
                    "Train a model using 'python src/training/train.py --config configs/e2_context_aware.yaml' "
                    "or configure MODEL_CHECKPOINT in backend/.env."
                ),
            },
        )

    try:
        # Run inference in worker thread pool with 60s timeout
        raw_poses = await asyncio.wait_for(
            run_in_threadpool(
                model_service.generate,
                text=req.text,
                context_poses=req.context_poses,
                max_frames=req.max_frames,
            ),
            timeout=60.0,
        )

        # Convert numpy array or tensors to list of lists of floats
        if hasattr(raw_poses, "tolist"):
            pose_list = raw_poses.tolist()
        elif hasattr(raw_poses, "numpy"):
            pose_list = raw_poses.numpy().tolist()
        else:
            pose_list = [list(f) for f in raw_poses]

        num_frames = len(pose_list)
        pose_dim = len(pose_list[0]) if num_frames > 0 else 274

        return GenerateResponse(
            poses=pose_list,
            num_frames=num_frames,
            pose_dim=pose_dim,
            model_ready=True,
            context_type=req.context_type,
            warning=RESEARCH_WARNING,
        )

    except ModelNotReadyError as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "ModelNotReady",
                "detail": str(e),
                "suggestion": "Ensure a valid PyTorch model checkpoint is trained and loaded.",
            },
        )
    except asyncio.TimeoutError:
        logger.error("Pose generation timed out after 60 seconds")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": "GenerationTimeout",
                "detail": "Pose generation exceeded the 60-second computation timeout.",
                "suggestion": "Reduce max_frames or use a shorter text prompt.",
            },
        )
    except Exception as e:
        logger.error("Unexpected error during pose generation: %s", e, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InferenceExecutionError",
                "detail": f"Generation failed unexpectedly: {str(e)}",
                "suggestion": "Check backend server logs for detailed traceback.",
            },
        )
