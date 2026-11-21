"""Canonical pose routes for the avatar renderer.

These routes are additive and work without any trained checkpoint. Model-backed
routes in backend/routers/*.py keep returning 503 until a checkpoint is loaded.

Wire format (``format=bin``):
    b"CSAP"                          4-byte magic
    uint32 little-endian version     1
    uint32 little-endian header_len
    header JSON (UTF-8)
    float32 little-endian payload    [T,137,2], row-major

See backend/services/canonical_pose.py::encode_bin.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.services.canonical_pose import (
    CanonicalSequence,
    adapt_model_output,
    adapt_repo_pose,
    encode_bin,
)
from backend.services.pose_demo_service import get_pose_demo_service

router = APIRouter(tags=["Avatar Poses"])


class AdaptRequest(BaseModel):
    points: Union[List[List[float]], List[List[List[float]]]] = Field(
        ..., description="[T,274] source-layout points, or [T,137,2]."
    )
    source_layout: str = "signlingo_openpose137"
    source: str = "generated"
    fps: float = 24.0


@router.get("/poses/demo")
def list_demo_sequences() -> dict:
    """List locally prepared ground-truth or synthetic demo sequences."""
    return get_pose_demo_service().list_demos()


@router.get("/poses/demo/{sequence_id}")
def get_demo_sequence(sequence_id: str, format: str = Query("json", pattern="^(json|bin)$")):
    return _respond(sequence_id, format)


@router.post("/poses/adapt")
def adapt_points(req: AdaptRequest) -> dict:
    """Adapt model or repository output to canonical space (returns JSON).

    Use this later to plug in generated ``[T,274]`` model output without
    touching the generation model. Points are treated as all-valid.
    """
    import numpy as np

    arr = np.asarray(req.points, dtype=np.float32)
    if req.source == "generated":
        seq = adapt_model_output(arr, fps=req.fps, source_layout=req.source_layout)
    else:
        seq = adapt_repo_pose(
            arr, source_layout=req.source_layout, source=req.source, fps=req.fps
        )
    return {
        "header": seq.header("uploaded"),
        "sentences": [],
        "points": seq.points.tolist(),
    }


@router.get("/poses/{sequence_id}")
def get_sequence(sequence_id: str, format: str = Query("json", pattern="^(json|bin)$")):
    """Return canonical keypoints as JSON or the CSAP binary format."""
    return _respond(sequence_id, format)


def _respond(sequence_id: str, format: str):
    service = get_pose_demo_service()
    try:
        seq: Optional[CanonicalSequence] = service.load(sequence_id)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if seq is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Sequence '{sequence_id}' not found. "
                "Run scripts/prepare_demo_poses.py or GET /api/poses/demo."
            ),
        )
    if format == "bin":
        body = encode_bin(seq, sequence_id)
        return Response(
            content=body,
            media_type="application/octet-stream",
            headers={
                "X-Pose-Format": "CSAP/1",
                "X-Pose-Sequence": sequence_id,
                "Cache-Control": "no-cache",
                "Access-Control-Expose-Headers": "X-Pose-Format, X-Pose-Sequence",
            },
        )
    return {
        "header": seq.header(sequence_id),
        "sentences": seq.meta.get("sentences", []),
        "meta": {k: v for k, v in seq.meta.items() if k != "sentences"},
        "points": seq.points.tolist(),
    }
