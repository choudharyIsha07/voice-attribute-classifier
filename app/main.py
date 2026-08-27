from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
from .config import settings
from .logging_config import setup_logging
from .api.routes import router
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from .api.routes import limiter
import numpy as np
setup_logging()

logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up librosa/numba to prevent JIT penalty on first request."""
    logger.info("Warming up inference provider...")
    from .services.inference import get_inference_provider
    provider = get_inference_provider()
    dummy_audio = np.zeros(16000, dtype=np.float32)
    provider.infer_attributes(dummy_audio, 16000)
    logger.info("Warmup complete.")
    yield

app = FastAPI(
    title=settings.app_name,
    description=(
        "Voice attribute inference service. Accepts audio uploads and returns "
        "estimated gender, age bracket, audio quality, and language. "
        "All processing is in-memory — no audio is persisted."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Internal error: {request.method} {request.url.path} — {type(exc).__name__}: {exc}"
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


app.include_router(router)
