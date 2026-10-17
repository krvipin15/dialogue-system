"""Dependencies Shared by the FastAPI Routes.

This module provides utility functions and FastAPI dependencies for
interacting with the LiveKit API, including credential validation
and URL transformation.
"""

from collections.abc import AsyncIterator

from fastapi import HTTPException
from livekit import api

from dialogue_system.config.logger import get_logger
from dialogue_system.config.settings import get_settings

# Initialize the logger
logger = get_logger(__name__)


async def get_livekit_api() -> AsyncIterator[api.LiveKitAPI]:
    """
    Yield a LiveKit API client and close its session after the request.

    This is a FastAPI dependency that manages the lifecycle of the LiveKit
    API client.

    Yields
    ------
    api.LiveKitAPI
        An initialized LiveKit API client.

    Raises
    ------
    Exception
        Logs and propagates any exceptions occurring during client
        instantiation or usage.
    """
    settings = get_settings()

    try:
        logger.debug("Initializing LiveKit API client")
        client = api.LiveKitAPI(
            url=settings.LIVEKIT_URL,
            api_key=settings.LIVEKIT_API_KEY.get_secret_value()
            if settings.LIVEKIT_API_KEY
            else None,
            api_secret=settings.LIVEKIT_API_SECRET.get_secret_value()
            if settings.LIVEKIT_API_SECRET
            else None,
        )
        yield client
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to initialize LiveKit API client: %s", e)
        raise
    finally:
        logger.debug("Closing LiveKit API client session")
        await client.aclose()
