"""Serve canonical demo pose sequences (no trained checkpoint required)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.services.canonical_pose import CanonicalSequence, NUM_POINTS, REPO_ROOT

DEFAULT_DIR = REPO_ROOT / "demo_data"


class PoseDemoService:
    def __init__(self, demo_dir: Optional[Path] = None) -> None:
        self.demo_dir = Path(
            demo_dir or os.environ.get("POSE_DEMO_DIR") or DEFAULT_DIR
        )
        self._lock = threading.Lock()
        self._manifest: Dict[str, Any] = {}
        self._manifest_mtime: Optional[float] = None

    # -- manifest ---------------------------------------------------------
    def _manifest_path(self) -> Path:
        return self.demo_dir / "manifest.json"

    def _load_manifest(self) -> Dict[str, Any]:
        path = self._manifest_path()
        if not path.exists():
            return {}
        mtime = path.stat().st_mtime
        with self._lock:
            if self._manifest_mtime != mtime:
                self._manifest = json.loads(path.read_text(encoding="utf-8"))
                self._manifest_mtime = mtime
            return self._manifest

    def _entry(self, sequence_id: str) -> Optional[Dict[str, Any]]:
        manifest = self._load_manifest()
        for entry in manifest.get("sequences", []):
            if entry.get("id") == sequence_id:
                return entry
        return None

    # -- public API -------------------------------------------------------
    def is_available(self) -> bool:
        return bool(self._load_manifest().get("sequences"))

    def list_demos(self) -> Dict[str, Any]:
        manifest = self._load_manifest()
        if not manifest:
            return {"count": 0, "sequences": [], "message": "No demo data prepared. Run scripts/prepare_demo_poses.py."}
        return {
            "count": len(manifest.get("sequences", [])),
            "defaultSequenceId": manifest.get("defaultSequenceId"),
            "sessionSequenceId": manifest.get("sessionSequenceId"),
            "fps": manifest.get("fps", 24.0),
            "layout": manifest.get("layout", "openpose137"),
            "realDataAvailable": manifest.get("realDataAvailable", False),
            "disclaimer": manifest.get("disclaimer", ""),
            "sequences": [
                {k: v for k, v in entry.items() if k != "file"}
                for entry in manifest.get("sequences", [])
            ],
        }

    def load(self, sequence_id: str) -> Optional[CanonicalSequence]:
        entry = self._entry(sequence_id)
        if entry is None:
            return None
        path = self.demo_dir / entry["file"]
        if not path.exists():
            return None
        points = np.load(path)
        if points.ndim != 3 or points.shape[1:] != (NUM_POINTS, 2):
            raise ValueError(f"Bad canonical array shape {points.shape} in {path}")
        manifest = self._load_manifest()
        return CanonicalSequence(
            points=points,
            fps=float(entry.get("fps", manifest.get("fps", 24.0))),
            layout=entry.get("layout", "openpose137"),
            normalization=entry.get("normalization", "neck_origin_shoulder_width"),
            source=entry.get("source", "synthetic"),
            meta={
                "title": entry.get("title", sequence_id),
                "synthetic": bool(entry.get("synthetic", False)),
                "warning": entry.get("warning", ""),
                "text": entry.get("text", ""),
                "sentences": entry.get("sentences", []),
                "disclaimer": manifest.get("disclaimer", ""),
            },
        )


_SERVICE: Optional[PoseDemoService] = None


def get_pose_demo_service() -> PoseDemoService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PoseDemoService()
    return _SERVICE
