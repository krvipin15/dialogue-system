"""FastAPI Application Entry Point.

This module initializes the FastAPI application, configures global middleware,
sets up the lifespan event handler for startup and shutdown logging, and
registers the API route handlers.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dialogue_system.api.routes import health, rooms
from dialogue_system.config.logger import configure_logging, get_logger
from dialogue_system.config.settings import get_settings

# Initialize settings and logging
settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    Run application startup and shutdown hooks.

    Yields
    ------
    None
    """
    try:
        logger.info("Starting %s environment=%s", settings.APP_NAME, settings.ENVIRONMENT.value)
        yield
    except Exception as e:
        logger.exception("An unexpected error occurred during the application lifespan: %s", e)
        raise
    finally:
        logger.info("Stopping API")


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Global exception handler to catch unhandled errors and log them
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for catching and logging unhandled server errors.

    Parameters
    ----------
    request : Request
        The incoming FastAPI request object.
    exc : Exception
        The exception instance that was raised.

    Returns
    -------
    JSONResponse
        A standardized JSON error response.
    """
    logger.error(
        "Unhandled exception occurred while processing %s %s: %s",
        request.method,
        request.url,
        str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


app.include_router(health.router, prefix="/api/v1")
app.include_router(rooms.router, prefix="/api/v1")


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """
    Expose service metadata and the health endpoint.

    Returns
    -------
    dict[str, str]
        A dictionary containing the application name and key endpoint URLs.
    """
    logger.debug("Root endpoint accessed")
    return {
        "name": settings.APP_NAME,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
