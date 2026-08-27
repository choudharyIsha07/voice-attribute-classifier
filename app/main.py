from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
from .config import settings
from .logging_config import setup_logging
from .api.routes import router

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Internal error processing request {request.method} {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

app.include_router(router)
