"""Argos Translation Module using the argostranslate Library.

This module provides a function to translate text using the
Argos Translate library. It loads the installed Argos language
packages once per process and reuses them for each translation
request, improving performance.
"""

from functools import lru_cache

from argostranslate import translate as argos_translate

from dialogue_system.config.logger import get_logger

# Initialize the logger
logger = get_logger(__name__)


@lru_cache(maxsize=64)
def _get_translation(source_language: str, target_language: str) -> argos_translate.ITranslation:
    """Get the Argos Translate translation object for the specified language pair.

    Parameters
    ----------
    source_language : str
        The language code of the source language.
    target_language : str
        The language code of the target language.

    Returns
    -------
    argos_translate.ITranslation
        The Argos Translate translation object for the specified language pair.
    """
    logger.debug("Looking up translation object for %s -> %s", source_language, target_language)
    installed_languages = argos_translate.get_installed_languages()

    # Find the source and target language objects from the installed languages
    source = next(
        (language for language in installed_languages if language.code == source_language), None
    )
    target = next(
        (language for language in installed_languages if language.code == target_language), None
    )

    # Check if either the source or target language is missing
    if source is None or target is None:
        missing = []
        if source is None:
            missing.append(source_language)
        if target is None:
            missing.append(target_language)
        logger.error("Argos language package missing for codes: %s", missing)
        raise RuntimeError(
            f"Argos language package missing: {source_language!r} -> {target_language!r}. "
            "Install the matching Argos Translate package before starting the call."
        )

    # Get the translation object for the source and target languages
    translation = source.get_translation(target)
    if translation is None:
        logger.error(
            "Argos translation direction not installed: %s -> %s", source_language, target_language
        )
        raise RuntimeError(
            f"Argos translation direction not installed: {source_language} -> {target_language}"
        )

    return translation


def translate_text(text: str, source_language: str, target_language: str) -> str:
    """Translate text from source language to target language.

    Parameters
    ----------
    text : str
        The text to be translated.
    source_language : str
        The language of the input text.
    target_language : str
        The language to translate the text into.

    Returns
    -------
    str
        The translated text.
    """
    # Strip whitespace from the input text and check if it's empty
    clean_text = text.strip()
    if not clean_text:
        logger.debug("Empty text provided for translation, returning as-is")
        return clean_text

    # Check if the source and target languages are the same
    if source_language == target_language:
        logger.info(
            "Source and target languages are identical (%s), skipping translation", source_language
        )
        return clean_text

    logger.info(
        "Translating text (%d chars) from %s to %s",
        len(clean_text),
        source_language,
        target_language,
    )

    # Get the translation object and perform the translation
    translated = _get_translation(source_language, target_language).translate(clean_text).strip()

    logger.debug(
        "Translated %s -> %s: '%s' -> '%s'",
        source_language,
        target_language,
        clean_text,
        translated,
    )
    return translated
