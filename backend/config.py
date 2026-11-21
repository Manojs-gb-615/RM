"""Configuration settings for ContextSign AI backend."""

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import field_validator

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=str(Path(__file__).resolve().parent / ".env"),
            env_file_encoding="utf-8",
            extra="ignore",
        )

        CHECKPOINT_DIR: str = "../checkpoints"
        DATA_DIR: str = "../data"
        CORS_ORIGINS: Union[List[str], str] = ["http://localhost:5173", "http://localhost:3000"]
        HOST: str = "0.0.0.0"
        PORT: int = 8000
        MODEL_CHECKPOINT: Optional[str] = None
        BASELINE_CHECKPOINT: Optional[str] = None
        CONTEXT_CHECKPOINT: Optional[str] = None

        @field_validator("CORS_ORIGINS", mode="before")
        @classmethod
        def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
            if isinstance(v, str):
                v_clean = v.strip()
                if v_clean.startswith("[") and v_clean.endswith("]"):
                    try:
                        return json.loads(v_clean)
                    except Exception:
                        pass
                return [s.strip() for s in v_clean.split(",") if s.strip()]
            return list(v)

        @field_validator("MODEL_CHECKPOINT", "BASELINE_CHECKPOINT", "CONTEXT_CHECKPOINT", mode="before")
        @classmethod
        def normalize_empty_to_none(cls, v: Optional[str]) -> Optional[str]:
            if isinstance(v, str) and not v.strip():
                return None
            return v

except ImportError:
    # pydantic v2 ships the v1 API; use it when pydantic-settings is absent.
    from pydantic.v1 import BaseSettings, validator  # type: ignore

    class Settings(BaseSettings):  # type: ignore
        class Config:
            env_file = str(Path(__file__).resolve().parent / ".env")
            env_file_encoding = "utf-8"
            extra = "ignore"

        CHECKPOINT_DIR: str = "../checkpoints"
        DATA_DIR: str = "../data"
        CORS_ORIGINS: Union[List[str], str] = ["http://localhost:5173", "http://localhost:3000"]
        HOST: str = "0.0.0.0"
        PORT: int = 8000
        MODEL_CHECKPOINT: Optional[str] = None
        BASELINE_CHECKPOINT: Optional[str] = None
        CONTEXT_CHECKPOINT: Optional[str] = None

        @validator("CORS_ORIGINS", pre=True)
        def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
            if isinstance(v, str):
                v_clean = v.strip()
                if v_clean.startswith("[") and v_clean.endswith("]"):
                    try:
                        return json.loads(v_clean)
                    except Exception:
                        pass
                return [s.strip() for s in v_clean.split(",") if s.strip()]
            return list(v)

        @validator("MODEL_CHECKPOINT", "BASELINE_CHECKPOINT", "CONTEXT_CHECKPOINT", pre=True)
        def normalize_empty_to_none(cls, v: Optional[str]) -> Optional[str]:
            if isinstance(v, str) and not v.strip():
                return None
            return v


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of Settings."""
    return Settings()
