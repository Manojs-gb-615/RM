"""Model management and inference service for ContextSign AI."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import get_settings

logger = logging.getLogger(__name__)


class ModelNotReadyError(Exception):
    """Raised when an inference operation is requested but no valid model checkpoint is loaded."""
    pass


# Add Research root to sys.path to enable imports from src/
_RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

# Attempt to import inference generator module dynamically
try:
    from src.inference.generator import PoseGenerator as _PoseGenerator
    from src.inference.generator import ModelNotReadyError as _InferenceModelNotReadyError
    PoseGenerator = _PoseGenerator
    ModelNotReadyError = _InferenceModelNotReadyError  # Use generator definition if available
except (ImportError, ModuleNotFoundError) as e:
    logger.info("PoseGenerator not loaded from src.inference (%s); will use fallback error handling", e)
    PoseGenerator = None


class ModelService:
    """Singleton service for loading model checkpoints and performing pose generation."""

    def __init__(self):
        self._ready: bool = False
        self.checkpoint_path: Optional[str] = None
        self.config_path: Optional[str] = None
        self.experiment_id: Optional[str] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.generator: Any = None

    def is_ready(self) -> bool:
        """Return True only if a valid model checkpoint is actively loaded and ready for inference."""
        return self._ready and (self.generator is not None)

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> bool:
        """Load a model checkpoint. Returns True if loaded successfully, False otherwise.

        Never crashes if checkpoint file does not exist.
        """
        settings = get_settings()
        resolved_ckpt = checkpoint_path or settings.MODEL_CHECKPOINT

        if not resolved_ckpt:
            logger.info("No checkpoint path provided or configured; model remains NOT_READY")
            self._ready = False
            self.checkpoint_path = None
            self.generator = None
            return False

        ckpt_file = Path(resolved_ckpt)
        if not ckpt_file.is_file():
            # Check relative to CHECKPOINT_DIR
            alt_ckpt = Path(settings.CHECKPOINT_DIR) / resolved_ckpt
            if alt_ckpt.is_file():
                ckpt_file = alt_ckpt
            else:
                logger.warning("Checkpoint file not found: %s; model remains NOT_READY", resolved_ckpt)
                self._ready = False
                self.checkpoint_path = None
                self.generator = None
                return False

        # If PoseGenerator class is not available
        if PoseGenerator is None:
            logger.warning("Inference generator module (src.inference.generator) is not available")
            self._ready = False
            return False

        try:
            logger.info("Loading model from checkpoint: %s", ckpt_file)
            self.generator = PoseGenerator(
                checkpoint_path=str(ckpt_file),
                config_path=config_path,
            )
            self._ready = True
            self.checkpoint_path = str(ckpt_file)
            self.config_path = config_path
            self.experiment_id = experiment_id or ckpt_file.stem.split("_")[0].upper()
            if hasattr(self.generator, "model_config"):
                self.model_config = self.generator.model_config
            logger.info("Model checkpoint successfully loaded from %s", ckpt_file)
            return True
        except Exception as e:
            logger.error("Failed to load model checkpoint %s: %s", ckpt_file, e, exc_info=True)
            self._ready = False
            self.generator = None
            self.checkpoint_path = None
            return False

    def generate(
        self,
        text: str,
        context_poses: Optional[Any] = None,
        max_frames: int = 150,
    ) -> Any:
        """Generate ASL poses for given text.

        Raises:
            ModelNotReadyError: If model is not loaded and ready.
        """
        if not self.is_ready():
            raise ModelNotReadyError(
                "Model is not ready: No checkpoint is loaded. "
                "ContextSign AI requires a trained model checkpoint for pose inference."
            )

        return self.generator.generate(
            text=text,
            context_poses=context_poses,
            max_frames=max_frames,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return model readiness and device status."""
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

        return {
            "ready": self.is_ready(),
            "checkpoint_path": self.checkpoint_path,
            "experiment_id": self.experiment_id,
            "model_config": self.model_config,
            "cuda_available": cuda_available,
        }


# Global singleton instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Return singleton instance of ModelService."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
