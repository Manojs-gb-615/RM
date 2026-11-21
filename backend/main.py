"""FastAPI application entrypoint for ContextSign AI backend."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.config import get_settings
from backend.routers import (
    architecture,
    compare,
    dataset,
    experiments,
    generate,
    health,
    model,
    poses,
)
from backend.schemas import ErrorResponse
from backend.services.model_service import ModelNotReadyError, get_model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("contextsign.backend")


def _dump_model(model_obj: Any) -> Dict[str, Any]:
    """Helper to dump Pydantic model for both v1 and v2."""
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown tasks."""
    settings = get_settings()
    logger.info("Initializing ContextSign AI API backend...")
    logger.info("Host: %s, Port: %d, CORS origins: %s", settings.HOST, settings.PORT, settings.CORS_ORIGINS)

    # Attempt to load checkpoint if configured
    model_service = get_model_service()
    if settings.MODEL_CHECKPOINT:
        logger.info("Configured MODEL_CHECKPOINT detected: %s", settings.MODEL_CHECKPOINT)
        loaded = model_service.load_model(settings.MODEL_CHECKPOINT)
        if loaded:
            logger.info("Startup: Checkpoint loaded successfully")
        else:
            logger.warning("Startup: Failed to load configured checkpoint; remaining NOT_READY")
    else:
        logger.info("Startup: No MODEL_CHECKPOINT configured; inference routes will return ModelNotReady")

    yield

    logger.info("ContextSign AI API backend shutting down...")


app = FastAPI(
    title="ContextSign AI API",
    description="Research-stage Context-Aware Sign Language Pose Generation API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

settings = get_settings()

# CORS configuration
origins = list(settings.CORS_ORIGINS) if isinstance(settings.CORS_ORIGINS, (list, tuple)) else ["*"]
# Ensure common local dev origins are included
for dev_origin in ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]:
    if dev_origin not in origins:
        origins.append(dev_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "%s %s -> %d (%.2f ms)",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
        return response
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            "%s %s -> EXCEPTION: %s (%.2f ms)",
            request.method,
            request.url.path,
            exc,
            latency_ms,
            exc_info=True,
        )
        raise


# Global exception handlers
@app.exception_handler(ModelNotReadyError)
async def handle_model_not_ready(request: Request, exc: ModelNotReadyError):
    resp = ErrorResponse(
        error="ModelNotReady",
        detail=str(exc),
        suggestion="Train a model checkpoint or configure MODEL_CHECKPOINT in backend/.env.",
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=_dump_model(resp),
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException):
    detail_str = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    resp = ErrorResponse(
        error=exc.__class__.__name__,
        detail=detail_str,
        suggestion="Verify request syntax and path parameters against the API documentation at /docs.",
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_dump_model(resp),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    detail_msg = "; ".join([f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in errors])
    resp = ErrorResponse(
        error="ValidationError",
        detail=detail_msg,
        suggestion="Ensure request payload matches the typed Pydantic schema required by this endpoint.",
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_dump_model(resp),
    )


@app.exception_handler(Exception)
async def handle_generic_exception(request: Request, exc: Exception):
    logger.error("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    resp = ErrorResponse(
        error="InternalServerError",
        detail=f"An unexpected internal server error occurred: {str(exc)}",
        suggestion="Check the server logs for detailed traceback information.",
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_dump_model(resp),
    )


# Root redirect
@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root path to interactive Swagger API documentation."""
    return RedirectResponse(url="/docs")


# Register API routers
app.include_router(health.router, prefix="/api")
app.include_router(model.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(compare.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(architecture.router, prefix="/api")
app.include_router(poses.router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
