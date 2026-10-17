"""LiveKit Agents worker entry point for the translation pipeline.

This module initializes the AgentServer, manages the prewarming of the
Silero VAD model to reduce cold-start latency, and defines the session
entrypoint that orchestrates the TranslationPipeline.
"""

import asyncio
from contextlib import suppress
from typing import Literal, cast

from livekit import agents
from livekit.agents import AgentServer, JobContext, JobProcess
from livekit.plugins import silero

from dialogue_system.agents.pipeline import TranslationPipeline
from dialogue_system.config.logger import configure_logging, get_logger
from dialogue_system.config.settings import get_settings

# Initialization
configure_logging()
logger = get_logger(__name__)
settings = get_settings()
server = AgentServer()

# Global set to prevent asyncio tasks from being garbage collected mid-execution
background_tasks = set()


def prewarm(proc: JobProcess) -> None:
    """
    Load Silero VAD once per worker process, before a room is assigned.

    Parameters
    ----------
    proc : JobProcess
        The worker process context where the VAD instance will be stored.
    """
    try:
        logger.info("Prewarming Silero VAD")
        proc.userdata["vad"] = silero.VAD.load(
            min_silence_duration=0.55,
            sample_rate=cast(Literal[8000, 16000], settings.WHISPER_SAMPLE_RATE),
        )
    except Exception as e:
        logger.exception("Failed to prewarm Silero VAD: %s", e)


server.setup_fnc = prewarm


@server.rtc_session(agent_name=settings.LIVEKIT_AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    """
    Connect to the assigned room and run the concurrent translation pipeline.

    This function manages the lifecycle of the translation agent, including
    connecting to the room, initializing the pipeline with the prewarmed VAD,
    and monitoring room occupancy to determine when to shut down.

    Parameters
    ----------
    ctx : JobContext
        The LiveKit job context providing access to the room and process data.

    Raises
    ------
    Exception
        Logs any unexpected errors during pipeline execution to prevent
        silent worker crashes.
    """
    try:
        await ctx.connect()
        logger.info("Translation worker connected room=%s", ctx.room.name)

        # Retrieve prewarmed VAD or load a fallback instance
        vad = ctx.proc.userdata.get("vad")
        if vad is None:
            logger.debug("VAD not found in userdata; loading fresh instance")
            vad = silero.VAD.load(
                min_silence_duration=0.55,
                sample_rate=cast(Literal[8000, 16000], settings.WHISPER_SAMPLE_RATE),
            )

        pipeline = TranslationPipeline(ctx.room, cast(silero.VAD, vad))
        pipeline.register_events()
        room_empty = asyncio.Event()

        async def stop_if_room_stays_empty() -> None:
            """Avoid shutting down during brief room membership update races."""
            await asyncio.sleep(1.0)
            if not ctx.room.remote_participants:
                logger.debug("Room is empty, triggering shutdown event")
                room_empty.set()

        def participant_disconnected(_participant: object) -> None:
            """Handle participant disconnection by checking for room emptiness."""
            task = asyncio.create_task(stop_if_room_stays_empty())
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

        ctx.room.on("participant_disconnected", participant_disconnected)

        # Process publications already subscribed before event registration completed.
        for participant in ctx.room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = publication.track
                if track is not None:
                    pipeline._on_track_subscribed(track, publication, participant)

        # Wait for participants to join or for the room to become empty
        if not ctx.room.remote_participants:
            logger.debug("Waiting for first participant to join...")
            await ctx.wait_for_participant()

        if ctx.room.remote_participants:
            await room_empty.wait()

    except Exception as e:
        logger.exception("Unexpected error in translation entrypoint: %s", e)
    finally:
        # Use suppress to handle cases where pipeline was never initialized
        with suppress(NameError):
            await pipeline.close()

        room_name = getattr(ctx, "room", "unknown")
        room_id = room_name.name if hasattr(room_name, "name") else room_name
        logger.info("Translation worker stopped room=%s", room_id)


if __name__ == "__main__":
    agents.cli.run_app(server)
