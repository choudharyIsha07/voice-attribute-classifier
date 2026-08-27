from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
from .config import settings
from .logging_config import setup_logging
from .api.routes import router

setup_logging()

logger = logging.getLogger(__name__)

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
)


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
