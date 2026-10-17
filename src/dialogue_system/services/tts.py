"""Piper TTS Module that Converts Text-to-Speech.

This module provides functionality to synthesize speech
from text using the Piper TTS engine. It supports multiple
languages and voices, and allows for configuration of output
sample rates and voice paths.
"""

import io
import math
import wave
from functools import lru_cache

import numpy as np
from piper import PiperVoice
from scipy.signal import resample_poly

from dialogue_system.config.logger import get_logger
from dialogue_system.config.settings import get_settings

# Initialize the logger
logger = get_logger(__name__)


@lru_cache(maxsize=8)
def _load_voice(language: str) -> PiperVoice:
    """Load the Piper voice for the specified language.

    Parameters
    ----------
    language : str
        The language code for which to load the Piper voice.

    Returns
    -------
    PiperVoice
        The loaded Piper voice object for the specified language.

    Raises
    ------
    FileNotFoundError
        If the Piper voice file for the specified language is not found.
    RuntimeError
        If the PiperVoice fails to load.
    KeyError
        If the language is not found in the configuration.
    """
    settings = get_settings()
    piper_voices = settings.PIPER_VOICES

    logger.debug("Attempting to load Piper voice for language: %s", language)

    if language not in piper_voices:
        logger.error(
            "Language '%s' not found in PIPER_VOICES configuration. Available: %s",
            language,
            list(piper_voices.keys()),
        )
        raise KeyError(f"Language '{language}' is not configured in PIPER_VOICES.")

    try:
        voice_config = piper_voices[language]
        logger.debug("Voice config for %s: %s", language, voice_config)

        # Extract parts and build path
        # Languages are in the format "en_male" or "en_female".
        voice_list = voice_config.split("/")[1:]
        filename = "-".join(voice_list) + ".onnx"
        path = settings.PIPER_DOWNLOAD_DIR / filename

        logger.debug("Constructed voice path: %s", path)

        if not path.is_file():
            logger.error("Voice file not found on disk: %s", path)
            raise FileNotFoundError(f"Piper voice file not found at {path}")

        logger.info("Loading Piper voice for language=%s from %s", language, path)
        return PiperVoice.load(str(path))

    except FileNotFoundError as e:
        raise e
    except Exception as e:
        logger.exception("Unexpected error loading Piper voice for %s: %s", language, str(e))
        raise RuntimeError(f"Failed to initialize PiperVoice for '{language}': {e}") from e


def synthesize_pcm(text: str, language: str) -> np.ndarray:
    """Synthesizes text to PCM audio data.

    Returns an empty array if text is empty. Returns an array of zeros
    if synthesis fails to prevent downstream system crashes.
    """
    settings = get_settings()

    if not text or not text.strip():
        logger.debug("Empty text provided for synthesis. Returning empty array.")
        return np.zeros(0, dtype=np.int16)

    try:
        logger.info("Synthesizing text [%s...] in language: %s", text[:20], language)
        voice = _load_voice(language)

        output = io.BytesIO()
        try:
            with wave.open(output, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
        except Exception as e:
            logger.exception("Piper synthesis engine failed for text: %s", text)
            raise RuntimeError(f"Piper synthesis error: {e}") from e

        output.seek(0)
        with wave.open(output, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            pcm_bytes = wav_file.readframes(wav_file.getnframes())

        logger.debug(
            "WAV properties: rate=%d, channels=%d, width=%d", sample_rate, channels, sample_width
        )

        width_count = 2
        if sample_width != width_count:
            logger.error("Unexpected sample width: %d. Expected: %d", sample_width, width_count)
            raise ValueError(f"Expected 16-bit Piper output, received {sample_width * 8}-bit audio")

        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).copy()

        if channels > 1:
            logger.debug("Downmixing %d channels to mono", channels)
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)

        if sample_rate != settings.PIPER_OUTPUT_SAMPLE_RATE:
            logger.debug(
                "Resampling audio from %dHz to %dHz", sample_rate, settings.PIPER_OUTPUT_SAMPLE_RATE
            )
            common = math.gcd(settings.PIPER_OUTPUT_SAMPLE_RATE, sample_rate)
            pcm = resample_poly(
                pcm.astype(np.float32),
                settings.PIPER_OUTPUT_SAMPLE_RATE // common,
                sample_rate // common,
            )
            pcm = np.clip(pcm, -32768, 32767).astype(np.int16)

        logger.info("Synthesis successful. PCM array size: %d", len(pcm))
        return pcm

    except (KeyError, FileNotFoundError) as e:
        logger.error("Configuration or File error during synthesis: %s", e)
        return np.zeros(1000, dtype=np.int16)
    except Exception as e:
        logger.exception("Critical error in synthesize_pcm: %s", str(e))
        return np.zeros(1000, dtype=np.int16)
