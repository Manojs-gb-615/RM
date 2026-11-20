"""
SignLingo AI Mini — FastAPI Backend
Run: uvicorn backend.main:app --reload --port 8000
"""

import os
import sys
import json
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from ml.inference import SignLingoInference
from ml.retrieval import parse_srt, parse_vtt, TranscriptSegment

# ── App setup ─────────────────────────────────────────────────────────────────

# ── Global inference engine ────────────────────────────────────────────────────

inference = SignLingoInference()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    data_dir = os.environ.get("DATA_DIR", "data")
    pose_dir = os.environ.get("POSE_DIR", None)
    inference.setup(data_dir=data_dir, pose_dir=pose_dir)
    yield


app = FastAPI(
    title="SignLingo AI Mini",
    description="Context-Aware ASL Pose Retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    current_sentence: str
    mode: str = "text_only"            # "text_only" | "context_aware"
    prev_sentence: Optional[str] = None
    prev_pose: Optional[List[List[float]]] = None   # [T, 274]

    @field_validator("mode")
    @classmethod
    def valid_mode(cls, v):
        if v not in ("text_only", "context_aware"):
            raise ValueError("mode must be 'text_only' or 'context_aware'")
        return v

    @field_validator("current_sentence")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("current_sentence must not be empty")
        return v.strip()


class CompareRequest(BaseModel):
    current_sentence: str
    prev_sentence: Optional[str] = None
    prev_pose: Optional[List[List[float]]] = None


class TranscriptSegmentOut(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    text: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "SignLingo AI Mini"}


@app.get("/api/status")
def status():
    return inference.status()


@app.get("/api/dataset/samples")
def dataset_samples():
    samples = inference.get_dataset_samples()
    return {
        "count": len(samples),
        "samples": samples,
    }


@app.get("/api/dataset/samples/{sample_id}/pose")
def get_pose(sample_id: str):
    pose = inference.get_sample_pose(sample_id)
    if pose is None:
        raise HTTPException(status_code=404, detail=f"Pose not found for sample '{sample_id}'")
    return {"sample_id": sample_id, "pose": pose, "frames": len(pose)}


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest):
    """
    Retrieve best matching ASL pose for a sentence.
    Returns pose array or no_match_reason if no reliable match.
    """
    result = inference.retrieve_for_sentence(
        current_sentence=req.current_sentence,
        mode=req.mode,
        prev_sentence=req.prev_sentence,
        prev_pose_list=req.prev_pose,
    )
    return result


@app.post("/api/compare")
def compare(req: CompareRequest):
    """
    Run both text_only and context_aware retrieval and return both results side by side.
    """
    text_only = inference.retrieve_for_sentence(
        current_sentence=req.current_sentence,
        mode="text_only",
    )
    context_aware = inference.retrieve_for_sentence(
        current_sentence=req.current_sentence,
        mode="context_aware",
        prev_sentence=req.prev_sentence,
        prev_pose_list=req.prev_pose,
    )
    return {
        "current_sentence": req.current_sentence,
        "text_only": text_only,
        "context_aware": context_aware,
        "same_result": text_only["matched_sample_id"] == context_aware["matched_sample_id"],
    }


@app.post("/api/transcript/parse")
async def parse_transcript(file: UploadFile = File(...)):
    """
    Upload an .srt or .vtt file, returns list of transcript segments.
    """
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in (".srt", ".vtt"):
        raise HTTPException(status_code=400, detail="Only .srt and .vtt files are supported.")

    # Read file (limit to 500 KB to prevent abuse)
    content_bytes = await file.read(500_000)
    try:
        content = content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode transcript: {e}")

    if ext == ".srt":
        segments = parse_srt(content)
    else:
        segments = parse_vtt(content)

    if not segments:
        raise HTTPException(status_code=422, detail="No transcript segments found in file.")

    return {
        "count": len(segments),
        "segments": [
            TranscriptSegmentOut(
                index=s.index,
                start_seconds=s.start_seconds,
                end_seconds=s.end_seconds,
                text=s.text,
            ).model_dump()
            for s in segments
        ],
    }


@app.post("/api/transcript/retrieve_all")
async def retrieve_all(
    transcript_file: UploadFile = File(...),
    mode: str = Form("context_aware"),
):
    """
    Parse a transcript and run retrieval for every segment.
    Returns all results in order (useful for pre-loading the full video session).
    """
    filename = transcript_file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in (".srt", ".vtt"):
        raise HTTPException(status_code=400, detail="Only .srt and .vtt files are supported.")
    content_bytes = await transcript_file.read(500_000)
    content = content_bytes.decode("utf-8", errors="replace")
    segments = parse_srt(content) if ext == ".srt" else parse_vtt(content)

    if not segments:
        raise HTTPException(status_code=422, detail="No transcript segments found.")

    results = []
    prev_sentence = None
    prev_pose = None

    for seg in segments:
        result = inference.retrieve_for_sentence(
            current_sentence=seg.text,
            mode=mode,
            prev_sentence=prev_sentence,
            prev_pose_list=prev_pose,
        )
        results.append({
            "segment_index": seg.index,
            "start_seconds": seg.start_seconds,
            "end_seconds": seg.end_seconds,
            "sentence": seg.text,
            "retrieval": result,
        })
        prev_sentence = seg.text
        prev_pose = result.get("pose")  # carry forward for next segment

    return {"mode": mode, "count": len(results), "results": results}


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )
