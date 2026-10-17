"""Speech-to-Text Module using faster-whisper.

This module provides functions to prepare audio and
transcribe it using the faster-whisper library. It
includes caching for the Whisper model to improve
performance on repeated calls.
"""

import time
from functools import lru_cache
from math import gcd

import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

from dialogue_system.config.logger import get_logger
from dialogue_system.config.settings import get_settings

# Initialize the logger
logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_whisper_model() -> WhisperModel:
    """Load and cache the Whisper model."""
    settings = get_settings()

    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s",
        settings.WHISPER_MODEL,
        settings.WHISPER_DEVICE,
        settings.WHISPER_COMPUTE_TYPE,
    )

    start = time.perf_counter()

    # Use the WhisperModel from faster-whisper with specified settings
    model = WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        download_root=str(settings.WHISPER_DOWNLOAD_DIR),
    )

    logger.debug("Whisper loaded in %.2fs", time.perf_counter() - start)
    return model


def prepare_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prepare audio for transcription.

    This function normalizes, resamples, and clips the audio to ensure
    it is in the correct format for the Whisper model.

    Parameters
    ----------
    audio : np.ndarray
        The input audio array (mono or stereo).
    sample_rate : int
        The sample rate of the input audio.

    Returns
    -------
    np.ndarray
        The processed audio array, ready for transcription.
    """
    settings = get_settings()

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    audio = np.asarray(audio)

    if audio.size == 0:
        return np.empty(0, dtype=np.float32)

    # SoundFile returns (samples, channels) for stereo audio.
    audio_count = 2
    if audio.ndim == audio_count:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio, got shape {audio.shape}")

    audio = audio.astype(np.float32, copy=False)

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite samples")

    peak = float(np.max(np.abs(audio)))

    logger.debug(
        "Input audio: rate=%d samples=%d duration=%.2fs peak=%.6f rms=%.6f",
        sample_rate,
        len(audio),
        len(audio) / sample_rate,
        peak,
        float(np.sqrt(np.mean(audio**2))),
    )

    # Check if the audio is silent or nearly silent.
    sound_threshold = 1e-7
    if peak < sound_threshold:
        logger.warning("Audio is silent or nearly silent")
        return np.empty(0, dtype=np.float32)

    # Resample rather than merely changing the sample-rate label.
    if sample_rate != settings.WHISPER_SAMPLE_RATE:
        divisor = gcd(sample_rate, settings.WHISPER_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            settings.WHISPER_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32)

        logger.info(
            "Resampled audio: %d Hz -> %d Hz",
            sample_rate,
            settings.WHISPER_SAMPLE_RATE,
        )

    # Clip rather than amplifying quiet background noise.
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def transcribe_pcm(audio: np.ndarray, language: str = "en") -> str:
    """Transcribe mono float32 audio sampled at 16 kHz."""
    settings = get_settings()

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError("transcribe_pcm expects mono audio")

    if audio.size == 0:
        logger.warning("No audio samples to transcribe")
        return ""

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid samples")

    logger.debug(
        "Transcribing: samples=%d duration=%.2fs language=%s",
        audio.size,
        audio.size / settings.WHISPER_SAMPLE_RATE,
        language,
    )

    model = _get_whisper_model()
    start = time.perf_counter()

    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        without_timestamps=True,
    )

    # Transcription runs lazily; consume the generator.
    result_segments = list(segments)
    elapsed = time.perf_counter() - start

    text = " ".join(
        segment.text.strip() for segment in result_segments if segment.text.strip()
    ).strip()

    logger.info(
        "Transcription finished in %.2fs; detected_language=%s probability=%.2f segments=%d",
        elapsed,
        info.language,
        info.language_probability,
        len(result_segments),
    )

    if not text:
        logger.warning(
            "No text detected. segments=%d duration=%.2fs audio_peak=%.6f",
            len(result_segments),
            len(audio) / settings.WHISPER_SAMPLE_RATE,
            float(np.max(np.abs(audio))),
        )

        for segment in result_segments:
            logger.debug(
                "Empty-result diagnostic: start=%.2f end=%.2f no_speech_prob=%.3f avg_logprob=%.3f",
                segment.start,
                segment.end,
                segment.no_speech_prob,
                segment.avg_logprob,
            )

    logger.debug("Transcribed text: %r", text)
    return text
