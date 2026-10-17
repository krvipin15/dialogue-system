"""Pydantic schemas for rooms, participants, and health responses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LanguageCode = Literal["en", "hi", "es", "fr", "de"]


class CreateRoomResponse(BaseModel):
    """Response returned after a room and agent dispatch are created."""

    model_config = ConfigDict(extra="forbid")

    room_name: str = Field(min_length=1, max_length=64)
    invite_url: str = Field(min_length=1)


class TokenRequest(BaseModel):
    """Request body used to create a participant access token."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=48)

    @field_validator("display_name")
    @classmethod
    def strip_display_name(cls, value: str) -> str:
        """Reject empty or whitespace-only participant names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Display name cannot be empty")
        return cleaned


class TokenResponse(BaseModel):
    """Room connection credentials returned to the browser."""

    model_config = ConfigDict(extra="forbid")

    token: str
    server_url: str
    room_name: str
    identity: str
    display_name: str


class HealthResponse(BaseModel):
    """Minimal health-check response."""

    status: Literal["ok"] = "ok"
    service: str = "dialogue-system-api"
