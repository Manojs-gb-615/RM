"""Pydantic request and response schemas for ContextSign AI API."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """System health status response."""
    status: str = Field(default="healthy", description="API health status")
    timestamp: str = Field(..., description="ISO 8601 current timestamp")
    model_ready: bool = Field(..., description="Whether model checkpoint is loaded and ready")
    cuda_available: bool = Field(..., description="Whether CUDA GPU acceleration is available")
    device: str = Field(..., description="Compute device name (e.g., cuda:0 or cpu)")


class ModelStatusResponse(BaseModel):
    """Detailed model status response."""

    model_config = ConfigDict(populate_by_name=True)

    ready: bool = Field(..., description="True if a valid model checkpoint is loaded")
    checkpoint_path: Optional[str] = Field(default=None, description="Path to active checkpoint file")
    experiment_id: Optional[str] = Field(default=None, description="Current experiment identifier")
    model_configuration: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="model_config",
        description="Model hyperparameter configuration",
    )
    cuda_available: bool = Field(default=False, description="Whether CUDA GPU is detected")


class GenerateRequest(BaseModel):
    """Sign pose generation request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Input English sentence to generate ASL poses for",
    )
    context_poses: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="Prior sentence ASL poses with shape [window, frames, pose_dim]",
    )
    context_type: Literal["none", "ground_truth", "generated"] = Field(
        default="none",
        description="Context conditioning mode",
    )
    max_frames: int = Field(
        default=150,
        ge=10,
        le=300,
        description="Maximum frames of ASL pose sequence to generate",
    )


class GenerateResponse(BaseModel):
    """Sign pose generation response."""
    poses: List[List[float]] = Field(
        ...,
        description="Generated 2D ASL pose sequence [frames, 274]",
    )
    num_frames: int = Field(..., description="Total generated frame count")
    pose_dim: int = Field(default=274, description="Pose dimension per frame (137 keypoints * 2)")
    model_ready: bool = Field(default=True, description="Whether inference model is active")
    context_type: str = Field(default="none", description="Context type used during generation")
    warning: Optional[str] = Field(
        default=None,
        description="Research disclaimer or model limitation warning",
    )


class CompareRequest(BaseModel):
    """Side-by-side comparison request between baseline and context-aware models."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Input English sentence",
    )
    context_poses: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="Optional prior pose sequence for context-aware model",
    )
    max_frames: int = Field(
        default=150,
        ge=10,
        le=300,
        description="Maximum frames to generate",
    )


class CompareResponse(BaseModel):
    """Comparison response contrasting baseline and context-aware outputs."""
    baseline: Optional[GenerateResponse] = Field(
        default=None,
        description="Output from baseline model (E1: text-only, no context)",
    )
    context_aware: Optional[GenerateResponse] = Field(
        default=None,
        description="Output from context-aware model (E2: with prior pose context)",
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Comparative metrics or status notes",
    )


class DatasetSample(BaseModel):
    """Single sample metadata from How2Sign dataset."""
    id: str = Field(..., description="Unique sample identifier (e.g., sentence name)")
    video_id: str = Field(..., description="Video ID the sample belongs to")
    sentence_id: str = Field(..., description="Sentence index or identifier")
    sentence: str = Field(..., description="English sentence transcript")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    prev_sentence: Optional[str] = Field(
        default=None,
        description="Previous sentence transcript in the same video context",
    )
    prev_sentence_id: Optional[str] = Field(
        default=None,
        description="ID of the previous sentence in the same video context",
    )
    split: str = Field(default="train", description="Dataset split (train, val, test)")


class DatasetSamplesResponse(BaseModel):
    """Paginated list of dataset samples."""
    samples: List[DatasetSample] = Field(..., description="List of dataset samples")
    total: int = Field(..., description="Total number of samples available in split")
    split: str = Field(..., description="Dataset split requested")


class ExperimentStatus(BaseModel):
    """Metadata and execution status for an experiment."""
    id: str = Field(..., description="Experiment ID (e.g., E1, E2, ...)")
    name: str = Field(..., description="Descriptive experiment title")
    status: Literal["IMPLEMENTED", "TESTED", "TRAINED", "EVALUATED", "PENDING"] = Field(
        ...,
        description="Current lifecycle status of the experiment",
    )
    description: str = Field(..., description="Summary of experiment purpose and design")
    config_path: Optional[str] = Field(
        default=None,
        description="Path to experiment YAML configuration file",
    )


class ExperimentsResponse(BaseModel):
    """List of all configured research experiments."""
    experiments: List[ExperimentStatus] = Field(..., description="List of experiments E1-E9")


class MetricsResponse(BaseModel):
    """Evaluation metrics for a specified experiment."""
    experiment_id: str = Field(..., description="Experiment ID queried")
    metrics: Dict[str, Any] = Field(..., description="Dictionary of metric values")
    data_type: Literal["synthetic_smoke_test", "real_how2sign", "not_evaluated"] = Field(
        ...,
        description="Provenance of the evaluation metrics",
    )
    timestamp: str = Field(..., description="Timestamp when metrics were recorded or queried")


class ArchitectureResponse(BaseModel):
    """Detailed architectural specifications of the ContextSign AI system."""

    model_config = ConfigDict(populate_by_name=True)

    components: List[Dict[str, Any]] = Field(..., description="List of architectural components")
    data_flow: List[str] = Field(..., description="Ordered steps of tensor data flow")
    model_configuration: Dict[str, Any] = Field(
        ..., alias="model_config", description="Model configuration parameters"
    )


class ErrorResponse(BaseModel):
    """Structured error response returned across API endpoints."""
    error: str = Field(..., description="High-level error classification or code")
    detail: str = Field(..., description="Human-readable explanation of what went wrong")
    suggestion: str = Field(..., description="Actionable recommendation for resolving the issue")
