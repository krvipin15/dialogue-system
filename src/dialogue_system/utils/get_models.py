"""Utility module for downloading and installing required models for the dialogue system.

This module provides functions to download and install Piper voice models,
Argos translation packages, and the Whisper model. It ensures that all necessary
models are available locally for the dialogue system to function correctly.
"""

from pathlib import Path

import requests
import requests.exceptions
from argostranslate import package as argos_package
from faster_whisper import WhisperModel

from dialogue_system.config.logger import get_logger
from dialogue_system.config.settings import get_settings

# Initialize the logger
logger = get_logger(__name__)


def _download_file(url: str, destination: Path) -> None:
    """Download a file from a URL to a local destination.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    destination : Path
        The local path where the file should be saved.

    Raises
    ------
    RuntimeError
        If the download fails or the file cannot be saved.
    """
    # Check if the file already exists and is of a reasonable size
    minimum_size = 1024  # 1 KB
    if destination.exists():
        file_size = destination.stat().st_size
        logger.debug("Checking existing file: %s (size: %d bytes)", destination, file_size)
        if file_size > minimum_size:
            logger.info("Already downloaded", destination=str(destination))
            return

    # Download the file with a timeout and stream it to avoid loading it all into memory
    logger.debug("Starting download from %s to %s", url, destination)
    try:
        with requests.get(url, timeout=90, stream=True) as response:
            response.raise_for_status()
            logger.debug(
                "Request successful. Content-Length: %s", response.headers.get("Content-Length")
            )

            with destination.open("wb") as file:
                bytes_downloaded = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    file.write(chunk)
                    bytes_downloaded += len(chunk)

        logger.debug("Successfully completed download to %s", destination)
    except requests.exceptions.RequestException as exc:
        logger.error("Download failed for %s: %s", url, exc)
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def download_piper_voices() -> None:
    """Download Piper voice models for the configured languages.

    This function iterates over the configured Piper voices and downloads the
    required model files (.onnx and .onnx.json) to the local Piper download directory.
    """
    settings = get_settings()
    logger.debug(
        "Initializing Piper voice downloads. Download directory: %s", settings.PIPER_DOWNLOAD_DIR
    )

    # Iterate over all required voices
    for voice_key, stem in settings.PIPER_VOICES.items():
        logger.info("Processing Piper voice", voice=voice_key, path=stem)

        # Split the stem to create a filename base
        parts = stem.split("/")
        filename_base = "-".join(parts[1:])
        logger.debug(
            "Derived filename base '%s' from stem '%s' for voice '%s'",
            filename_base,
            stem,
            voice_key,
        )

        # Piper models require both the .onnx and the .onnx.json config file
        for suffix in (".onnx", ".onnx.json"):
            destination = settings.PIPER_DOWNLOAD_DIR / f"{filename_base}{suffix}"
            download_url = (
                f"{settings.PIPER_VOICES_BASE_URL}{stem}/{filename_base}{suffix}?download=true"
            )

            logger.debug("Attempting to download %s from %s", suffix, download_url)

            try:
                _download_file(download_url, destination)
                logger.debug("Successfully processed %s for voice %s", suffix, voice_key)
            except requests.exceptions.RequestException as exc:
                logger.error(
                    "Failed to download Piper voice file",
                    voice=voice_key,
                    file=f"{filename_base}{suffix}",
                    error=str(exc),
                )


def install_argos_pairs(pairs: list[tuple[str, str]]) -> None:
    """Install Argos translation packages for the specified language pairs.

    Parameters
    ----------
    pairs : list[tuple[str, str]]
        A list of tuples where each tuple contains a source and target language code.
    """
    settings = get_settings()

    # Update the Argos package index and retrieve available packages
    logger.info("Updating Argos package index")
    argos_package.update_package_index()
    packages = argos_package.get_available_packages()
    logger.debug("Retrieved %d available Argos packages from index", len(packages))

    for source, target in pairs:
        logger.debug("Searching for Argos pair: %s -> %s", source, target)

        # Find the first matching package for the source-target pair
        candidate = next(
            (item for item in packages if item.from_code == source and item.to_code == target),
            None,
        )

        # Continue to the next pair if no candidate is found
        if candidate is None:
            logger.warning(
                "Argos package not available in current index", source=source, target=target
            )
            continue

        # Download and install the candidate package
        logger.debug("Candidate found: %s. Starting download...", candidate)
        downloaded = candidate.download()
        downloaded_path = Path(downloaded)
        local_copy = settings.ARGOS_DOWNLOAD_DIR / downloaded_path.name
        logger.debug("Moving downloaded file from %s to %s", downloaded_path, local_copy)
        local_copy.write_bytes(downloaded_path.read_bytes())
        logger.info("Installing Argos package from local path: %s", local_copy)
        argos_package.install_from_path(local_copy)
        logger.debug("Successfully installed pair %s -> %s", source, target)


def main() -> None:
    """Entry point function to download and install all required models."""
    settings = get_settings()

    logger.info("Downloading Piper voice models...")
    download_piper_voices()

    logger.info("Downloading Argos language pairs...")
    codes = list(dict.fromkeys(settings.MODEL_LANGUAGES))
    pairs = [(source, target) for source in codes for target in codes if source != target]
    install_argos_pairs(pairs)

    logger.info("Downloading Whisper model...")
    WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        download_root=str(settings.WHISPER_DOWNLOAD_DIR),
    )

    logger.info("All models downloaded successfully.")


if __name__ == "__main__":
    main()
