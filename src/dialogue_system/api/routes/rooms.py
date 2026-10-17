"""Room creation, invitation, participant token, and agent dispatch routes.

Creates LiveKit rooms, dispatches the translation agent, and issues
short-lived JWT access tokens for browser participants.
"""

import logging
from datetime import timedelta
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from livekit import api
from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest
from livekit.protocol.room import CreateRoomRequest

from dialogue_system.api.dependencies import get_livekit_api
from dialogue_system.config.settings import Settings, get_settings
from dialogue_system.schemas.room import (
    CreateRoomResponse,
    TokenRequest,
    TokenResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rooms", tags=["rooms"])

LiveKitClient = Annotated[api.LiveKitAPI, Depends(get_livekit_api)]


def _require_livekit_setting(value: object | None, name: str) -> str:
    """Validate a required LiveKit setting and unwrap SecretStr values."""
    if value is None:
        logger.error("Configuration error: %s is missing", name)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LiveKit is not configured: {name} is missing.",
        )

    getter = getattr(value, "get_secret_value", None)
    result = getter() if callable(getter) else str(value)

    if not result.strip():
        logger.error("Configuration error: %s is empty", name)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LiveKit is not configured: {name} is empty.",
        )

    return result


def _server_url_for_browser(settings: Settings | None = None) -> str:
    """Convert the configured LiveKit URL to a browser WebSocket URL."""
    config = settings or get_settings()
    raw_url = config.LIVEKIT_URL

    if not raw_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LiveKit is not configured: LIVEKIT_URL is missing.",
        )

    url = raw_url.strip().rstrip("/")

    if url.startswith("ws://livekit:"):
        return url.replace("ws://livekit:", "ws://localhost:", 1)

    if url.startswith("http://livekit:"):
        return url.replace("http://livekit:", "ws://localhost:", 1)

    if url.startswith(("ws://", "wss://")):
        return url

    if url.startswith("https://"):
        return "wss://" + url.removeprefix("https://")

    if url.startswith("http://"):
        return "ws://" + url.removeprefix("http://")

    logger.error("Unsupported LIVEKIT_URL protocol")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="LIVEKIT_URL must use ws://, wss://, http://, or https://.",
    )


@router.post(
    "",
    response_model=CreateRoomResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_room(client: LiveKitClient) -> CreateRoomResponse:
    """Create a LiveKit room, dispatch the translation agent, and return the invitation URL.

    FastAPI manages the LiveKit client's lifecycle through the dependency.
    Do not manually close the client in this route.
    """
    settings = get_settings()

    agent_name = _require_livekit_setting(
        settings.LIVEKIT_AGENT_NAME,
        "LIVEKIT_AGENT_NAME",
    )
    frontend_url = _require_livekit_setting(
        settings.PUBLIC_FRONTEND_URL,
        "PUBLIC_FRONTEND_URL",
    )

    room_name = f"dialogue-{uuid4().hex[:12]}"

    try:
        logger.info("Creating LiveKit room: %s", room_name)

        await client.room.create_room(
            CreateRoomRequest(
                name=room_name,
                empty_timeout=300,
                max_participants=3,
            )
        )

        logger.info(
            "Dispatching agent %s to room %s",
            agent_name,
            room_name,
        )

        await client.agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=room_name,
            )
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to create room or dispatch agent for %s",
            room_name,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Could not create the LiveKit room or dispatch the "
                "translation agent. Verify LiveKit connectivity, "
                "credentials, and agent worker status."
            ),
        ) from exc

    invite_url = f"{frontend_url.rstrip('/')}/room/{room_name}"

    logger.info("Room created successfully: %s", room_name)

    return CreateRoomResponse(
        room_name=room_name,
        invite_url=invite_url,
    )


@router.post(
    "/{room_name}/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
)
async def create_participant_token(
    room_name: str,
    payload: TokenRequest,
) -> TokenResponse:
    """
    Generate a one-hour, room-scoped JWT for a browser participant.

    The token grants permission to join the room, publish and subscribe
    to media, and publish data messages.
    """
    max_room = 64
    if (
        not room_name.startswith("dialogue-")
        or len(room_name) > max_room
        or len(room_name) <= len("dialogue-")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid room name.",
        )

    settings = get_settings()

    api_key = _require_livekit_setting(
        settings.LIVEKIT_API_KEY,
        "LIVEKIT_API_KEY",
    )
    api_secret = _require_livekit_setting(
        settings.LIVEKIT_API_SECRET,
        "LIVEKIT_API_SECRET",
    )

    identity = f"guest-{uuid4().hex[:10]}"

    try:
        access_token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_name(payload.display_name)
            .with_ttl(timedelta(hours=1))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
        )

        jwt_token = access_token.to_jwt()

    except Exception as exc:
        logger.exception(
            "Failed to generate participant token for %s",
            identity,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create a participant access token.",
        ) from exc

    return TokenResponse(
        token=jwt_token,
        server_url=_server_url_for_browser(settings),
        room_name=room_name,
        identity=identity,
        display_name=payload.display_name,
    )
