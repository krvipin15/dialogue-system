"""Health Endpoint.

This module provides a simple health check route used by orchestrators
or monitoring tools to verify that the service process is running.
"""

import logging

from fastapi import APIRouter

from dialogue_system.schemas.room import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Return a lightweight process health response.

    Returns
    -------
    HealthResponse
        A schema indicating the current health status of the application.
    """
    logger.debug("Health check endpoint accessed")
    try:
        return HealthResponse()
    except Exception as e:
        logger.error("Unexpected error while generating health response: %s", e)
        raise
