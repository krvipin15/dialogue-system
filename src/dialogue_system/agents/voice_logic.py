"""Audio frame conversion and barge-in helpers for LiveKit audio streams.

This module provides utility functions to transform LiveKit AudioFrame
sequences into NumPy arrays suitable for ML models (float32) or
diagnostics (int16), while handling channel mixing and utterance length limits.
"""

import inspect

import numpy as np
from livekit import rtc

from dialogue_system.config.settings import get_settings


def frames_to_float32(frames: list[rtc.AudioFrame]) -> np.ndarray:
    """
    Convert VAD frames into contiguous mono float32 samples in [-1, 1].

    This function processes a list of audio frames, handles multi-channel
    audio by averaging channels into mono, and clips the resulting
    audio to the maximum utterance duration defined in settings.

    Parameters
    ----------
    frames : list[rtc.AudioFrame]
        A list of LiveKit audio frames to be converted.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of float32 samples normalized between -1.0 and 1.0.
    """
    settings = get_settings()

    chunks: list[np.ndarray] = []
    for frame in frames:
        raw = frame.data
        samples = (
            np.asarray(raw, dtype=np.int16).reshape(-1)
            if isinstance(raw, np.ndarray)
            else np.frombuffer(raw, dtype=np.int16)
        )
        if frame.num_channels > 1:
            usable = samples.size - samples.size % frame.num_channels
            if usable == 0:
                continue
            samples = (
                samples[:usable].reshape(-1, frame.num_channels).astype(np.float32).mean(axis=1)
            )
        chunks.append(np.asarray(samples, dtype=np.float32) / 32768.0)

    if not chunks:
        return np.empty(0, dtype=np.float32)

    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    max_seconds = settings.MAX_UTTERANCE_DURATION
    max_samples = int(settings.WHISPER_SAMPLE_RATE * max_seconds)
    if max_samples > 0 and audio.size > max_samples:
        audio = audio[-max_samples:]
    return np.clip(audio, -1.0, 1.0)


def frames_to_int16(frames: list[rtc.AudioFrame]) -> np.ndarray:
    """
    Convert frames to mono int16 PCM, mainly for tests and diagnostics.

    Parameters
    ----------
    frames : list[rtc.AudioFrame]
        A list of LiveKit audio frames to be converted.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of int16 samples.
    """
    audio = frames_to_float32(frames)
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)


async def clear_audio_queue(source: rtc.AudioSource) -> None:
    """
    Clear a LiveKit audio source queue when supported by the installed SDK.

    This is typically used during barge-in events to discard stale audio
    data before starting a new processing cycle.

    Parameters
    ----------
    source : rtc.AudioSource
        The LiveKit audio source whose queue should be cleared.
    """
    clear_queue = getattr(source, "clear_queue", None)
    if clear_queue is None:
        return
    result = clear_queue()
    if inspect.isawaitable(result):
        await result
