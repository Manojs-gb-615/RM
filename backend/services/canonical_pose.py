"""Canonical pose adapter for the ContextSign avatar renderer.

Converts repository pose arrays into the canonical avatar-input space:

    origin = neck (body index 1)
    unit   = per-clip median shoulder width (indices 2 and 5)
    +x     = signer's anatomical left in the image
    +y     = up
    +z     = toward the camera (depth is inferred in the browser)

The repository's actual layout and normalization differ from the task's
canonical layout; both are declared in ``layouts/*.json`` (see DECISIONS.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

POSE_DIM = 274
NUM_POINTS = 137

REPO_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_DIR = REPO_ROOT / "layouts"

_MAGIC = b"CSAP"
_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

def load_layout(name: str = "openpose137") -> Dict[str, Any]:
    path = LAYOUT_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Layout not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def source_to_canonical(points: np.ndarray, source_layout: str = "signlingo_openpose137") -> np.ndarray:
    """Reorder ``[..., 137, 2]`` points from ``source_layout`` to canonical order."""
    if source_layout == "openpose137":
        return points
    layout = load_layout(source_layout)
    mapping = layout["sourceToCanonical"]
    if len(mapping) != NUM_POINTS:
        raise ValueError(f"Layout {source_layout} has {len(mapping)} points, expected {NUM_POINTS}")
    return points[..., mapping, :]


def canonical_to_source(points: np.ndarray, source_layout: str = "signlingo_openpose137") -> np.ndarray:
    if source_layout == "openpose137":
        return points
    layout = load_layout(source_layout)
    mapping = layout["canonicalToSource"]
    return points[..., mapping, :]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CanonicalSequence:
    points: np.ndarray                 # [T, 137, 2] float32, canonical units
    fps: float = 24.0
    layout: str = "openpose137"
    normalization: str = "neck_origin_shoulder_width"
    source: str = "synthetic"          # "real" | "synthetic" | "generated"
    valid: Optional[np.ndarray] = None  # [T, 137] bool, after gap filling all True
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 3 or self.points.shape[1:] != (NUM_POINTS, 2):
            raise ValueError(
                f"Canonical points must be [T,{NUM_POINTS},2]; got {self.points.shape}"
            )
        if not np.isfinite(self.points).all():
            self.points = np.nan_to_num(self.points, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.valid is None:
            self.valid = np.ones((self.points.shape[0], NUM_POINTS), dtype=bool)

    @property
    def num_frames(self) -> int:
        return int(self.points.shape[0])

    def header(self, sequence_id: str = "") -> Dict[str, Any]:
        return {
            "format": "CSAP",
            "version": _FORMAT_VERSION,
            "sequenceId": sequence_id,
            "fps": float(self.fps),
            "layout": self.layout,
            "T": self.num_frames,
            "points": NUM_POINTS,
            "dims": 2,
            "dtype": "float32",
            "endianness": "little",
            "normalization": self.normalization,
            "source": self.source,
            **({k: v for k, v in self.meta.items() if k != "sentences"}),
        }


# ---------------------------------------------------------------------------
# Normalization inversion
# ---------------------------------------------------------------------------

def _median_shoulder_width(points: np.ndarray, valid: np.ndarray) -> float:
    """Median pixel/unit distance between the shoulders across valid frames."""
    widths = []
    for t in range(points.shape[0]):
        if valid[t, 2] and valid[t, 5]:
            d = float(np.linalg.norm(points[t, 2] - points[t, 5]))
            if d > 1e-6:
                widths.append(d)
    if not widths:
        return 1.0
    return float(np.median(widths))


def invert_repo_normalization(
    points: np.ndarray,
    *,
    source_width: float = 1280.0,
    source_height: float = 720.0,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert ``ml/convert_openpose.py`` normalization on canonical-ordered points.

    The converter writes ``x = px / width`` and ``y = py / height`` clipped to
    ``[0, 1]`` (y down). The optional ``(x - mean) / std`` step from
    ``src/data/how2sign_dataset.py`` is undone first when stats are supplied.
    Returns pixel-space points (y up) and the raw missing mask.
    """
    pts = np.asarray(points, dtype=np.float64).copy()
    missing = (np.abs(pts[..., 0]) < 1e-9) & (np.abs(pts[..., 1]) < 1e-9)

    if norm_mean is not None and norm_std is not None:
        mean = np.asarray(norm_mean, dtype=np.float64).reshape(NUM_POINTS, 2)
        std = np.asarray(norm_std, dtype=np.float64).reshape(NUM_POINTS, 2)
        pts = pts * std + mean

    px = pts[..., 0] * float(source_width)
    py = (1.0 - pts[..., 1]) * float(source_height)
    pixels = np.stack([px, py], axis=-1)
    return pixels, missing


def canonicalize_pixels(
    pixels: np.ndarray,
    missing: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Centre on the neck and divide by median shoulder width (pixel space)."""
    valid = ~missing
    width = _median_shoulder_width(pixels, valid)
    neck = np.zeros((pixels.shape[0], 1, 2), dtype=np.float64)
    # Median neck position over valid frames keeps the origin stable without
    # per-frame jitter; per-frame x/y detail is preserved relative to it.
    neck_y = np.median(pixels[valid[:, 1], 1, 1]) if valid[:, 1].any() else 0.0
    neck_x = np.median(pixels[valid[:, 1], 1, 0]) if valid[:, 1].any() else 0.0
    neck[:, 0, 0] = neck_x
    neck[:, 0, 1] = neck_y
    out = (pixels - neck) / max(width, 1e-6)
    out[missing] = 0.0
    return out.astype(np.float32), valid


# ---------------------------------------------------------------------------
# Missing-point repair
# ---------------------------------------------------------------------------

def fill_missing(
    points: np.ndarray,
    valid: np.ndarray,
    max_gap: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate short gaps, hold the nearest valid value for long ones.

    ``points`` is ``[T, 137, 2]`` and ``valid`` is ``[T, 137]``. Every point is
    repaired independently. Returns repaired points and a mask of points that
    were repaired (used for confidence reporting, not for validity gating).
    """
    out = np.asarray(points, dtype=np.float64).copy()
    repaired = (~valid) & np.zeros_like(valid, dtype=bool)
    T = out.shape[0]

    for k in range(out.shape[1]):
        ok = valid[:, k]
        if ok.all():
            continue
        if not ok.any():
            out[:, k] = 0.0
            continue
        idx = np.flatnonzero(ok)
        missing_idx = np.flatnonzero(~ok)
        repaired[missing_idx, k] = True
        for t in missing_idx:
            prev = idx[idx < t]
            nxt = idx[idx > t]
            if prev.size and nxt.size and (nxt[0] - prev[-1] - 1) <= max_gap:
                a, b = int(prev[-1]), int(nxt[0])
                w = (t - a) / (b - a)
                out[t, k] = (1.0 - w) * out[a, k] + w * out[b, k]
            elif prev.size:
                out[t, k] = out[prev[-1], k]
            else:
                out[t, k] = out[nxt[0], k]
    return out.astype(np.float32), repaired


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def adapt_repo_pose(
    raw: np.ndarray,
    *,
    source_layout: str = "signlingo_openpose137",
    source: str = "real",
    fps: float = 24.0,
    source_width: float = 1280.0,
    source_height: float = 720.0,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    max_gap: int = 5,
) -> CanonicalSequence:
    """Adapt a repository pose array to canonical space.

    Accepts ``[T, 274]`` (flat), ``[T, 137, 2]`` in the source layout, or
    canonical ``[T, 137, 2]`` (``source_layout="openpose137"``). Generated
    output must pass ``source="generated"`` so no point is treated as missing.
    """
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == POSE_DIM:
        arr = arr.reshape(arr.shape[0], NUM_POINTS, 2)
    if arr.ndim != 3 or arr.shape[1:] != (NUM_POINTS, 2):
        raise ValueError(f"Expected [T,274] or [T,137,2]; got {arr.shape}")
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    arr = source_to_canonical(arr, source_layout)

    if source_layout == "openpose137":
        # Already in canonical *coordinate* space; only the sentinel matters.
        missing = (np.abs(arr[..., 0]) < 1e-9) & (np.abs(arr[..., 1]) < 1e-9)
        pixels = arr.astype(np.float64)
    else:
        pixels, missing = invert_repo_normalization(
            arr,
            source_width=source_width,
            source_height=source_height,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    if source == "generated":
        # Model output has no confidence channel; treat all points as valid.
        missing = np.zeros_like(missing, dtype=bool)

    # Centre on the neck and normalise by median shoulder width. This is a
    # no-op-scale operation for canonical input (shoulder width is already 1).
    points, valid = canonicalize_pixels(pixels, missing)

    repaired = np.zeros_like(valid, dtype=bool)
    if (~valid).any():
        points, repaired = fill_missing(points, valid, max_gap=max_gap)
        valid = valid | repaired

    return CanonicalSequence(
        points=points,
        fps=float(fps),
        layout="openpose137",
        normalization="neck_origin_shoulder_width",
        source=source,
        valid=valid,
        meta={
            "sourceLayout": source_layout,
            "repairedPoints": int(repaired.sum()),
            "missingPoints": int((~np.asarray(valid)).sum()),
        },
    )


def adapt_model_output(
    raw: np.ndarray,
    *,
    fps: float = 24.0,
    source_layout: str = "signlingo_openpose137",
) -> CanonicalSequence:
    """Adapt ``[B,T,274]`` model output (batch 1) with all points valid."""
    arr = np.asarray(raw)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError("adapt_model_output accepts batch size 1")
        arr = arr[0]
    return adapt_repo_pose(arr, source_layout=source_layout, source="generated", fps=fps)


# ---------------------------------------------------------------------------
# Wire format: CSAP header + float32 payload
# ---------------------------------------------------------------------------

def encode_bin(sequence: CanonicalSequence, sequence_id: str = "") -> bytes:
    header = json.dumps(sequence.header(sequence_id), separators=(",", ":")).encode("utf-8")
    payload = np.ascontiguousarray(sequence.points, dtype="<f4").tobytes()
    return _MAGIC + int(_FORMAT_VERSION).to_bytes(4, "little") + len(header).to_bytes(4, "little") + header + payload


def decode_bin(blob: bytes) -> Tuple[Dict[str, Any], np.ndarray]:
    if len(blob) < 12 or blob[:4] != _MAGIC:
        raise ValueError("Not a CSAP binary pose payload")
    version = int.from_bytes(blob[4:8], "little")
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported CSAP version {version}")
    header_len = int.from_bytes(blob[8:12], "little")
    header = json.loads(blob[12:12 + header_len].decode("utf-8"))
    payload = blob[12 + header_len:]
    T = int(header["T"])
    expected = T * NUM_POINTS * 2 * 4
    if len(payload) != expected:
        raise ValueError(f"Payload length {len(payload)} != expected {expected}")
    arr = np.frombuffer(payload, dtype="<f4").reshape(T, NUM_POINTS, 2).astype(np.float32)
    return header, arr
