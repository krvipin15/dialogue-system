"""Pipeline for handling audio processing and translation in a LiveKit room.

This module defines the `TranslationPipeline` class, which
manages the lifecycle of audio tracks, transcription, translation,
and text-to-speech synthesis for participants in a LiveKit room.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from livekit import rtc
from livekit.agents import vad as agents_vad
from livekit.plugins import silero

from dialogue_system.agents.voice_logic import clear_audio_queue, frames_to_float32
from dialogue_system.config.logger import get_logger
from dialogue_system.config.settings import get_settings
from dialogue_system.services.stt import transcribe_pcm
from dialogue_system.services.translation import translate_text
from dialogue_system.services.tts import synthesize_pcm

# Initialize the logger
logger = get_logger(__name__)


@dataclass(slots=True)
class PublishedVoice:
    """A long-lived LiveKit output track used for translated speech."""

    source: rtc.AudioSource
    track: rtc.LocalAudioTrack
    publication: Any


class TranslationPipeline:
    """Listen to each human's track and independently translate their speech."""

    def __init__(self, room: rtc.Room, vad: silero.VAD) -> None:
        """Initialize the translation pipeline with a LiveKit room and a prewarmed Silero VAD instance.

        Parameters
        ----------
        room : rtc.Room
            The LiveKit room instance.
        vad : silero.VAD
            The prewarmed Silero VAD instance.
        """
        self.room = room
        self.vad = vad
        self._track_tasks: dict[str, asyncio.Task[None]] = {}
        self._utterance_tasks: set[asyncio.Task[None]] = set()
        self._output_tracks: dict[tuple[str, str], PublishedVoice] = {}
        self._playback_tasks: dict[str, set[asyncio.Task[None]]] = {}
        self._transcription_semaphore = asyncio.Semaphore(2)
        self._closed = False

    def register_events(self) -> None:
        """Subscribe to room track lifecycle events."""
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        """Start consuming a new audio track if it's from a human participant.

        Parameters
        ----------
        track : rtc.Track
            The newly subscribed track.
        publication : rtc.RemoteTrackPublication
            The publication metadata for the track.
        participant : rtc.RemoteParticipant
            The participant who published the track.
        """
        del publication
        if self._closed or track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity.startswith("translator-"):
            return
        key = track.sid
        if key in self._track_tasks:
            return
        task = asyncio.create_task(self._consume_audio(track, participant), name=f"audio-{key}")
        self._track_tasks[key] = task
        logger.info("Listening to participant=%s track=%s", participant.identity, key)

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        """Stop consuming an audio track when it is unsubscribed.

        Parameters
        ----------
        track : rtc.Track
            The unsubscribed track.
        publication : rtc.RemoteTrackPublication
            The publication metadata for the track.
        participant : rtc.RemoteParticipant
            The participant who published the track.
        """
        del publication, participant
        task = self._track_tasks.pop(track.sid, None)
        if task is not None:
            task.cancel()

    async def _consume_audio(self, track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        """Pass a remote audio track through a dedicated Silero VAD stream.

        Parameters
        ----------
        track : rtc.Track
            The remote audio track to consume.
        participant : rtc.RemoteParticipant
            The participant owning the track.
        """
        vad_stream = self.vad.stream()

        async def read_vad_events() -> None:
            async for event in vad_stream:
                if event.type == agents_vad.VADEventType.START_OF_SPEECH:
                    await self._interrupt_listener(participant.identity)
                elif event.type == agents_vad.VADEventType.END_OF_SPEECH and event.frames:
                    task = asyncio.create_task(
                        self._handle_utterance(participant, list(event.frames)),
                        name=f"utterance-{participant.identity}-{time.monotonic_ns()}",
                    )
                    self._utterance_tasks.add(task)
                    task.add_done_callback(self._utterance_tasks.discard)

        event_task = asyncio.create_task(read_vad_events())
        try:
            stream = rtc.AudioStream(
                track, sample_rate=get_settings().WHISPER_SAMPLE_RATE, num_channels=1
            )
            async for event in stream:
                vad_stream.push_frame(event.frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Audio stream failed for participant=%s", participant.identity)
        finally:
            event_task.cancel()
            await asyncio.gather(event_task, return_exceptions=True)
            close = getattr(vad_stream, "aclose", None)
            if close is not None:
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("Failed to close VAD stream cleanly")

    async def _interrupt_listener(self, listener_identity: str) -> None:
        """Cancel translated playback queued for a participant who starts speaking.

        Parameters
        ----------
        listener_identity : str
            The identity of the participant who started speaking.
        """
        for task in tuple(self._playback_tasks.get(listener_identity, set())):
            if not task.done():
                task.cancel()

        for (speaker_identity, _language), output in self._output_tracks.items():
            if speaker_identity != listener_identity:
                try:
                    await clear_audio_queue(output.source)
                except Exception:
                    logger.exception("Failed to clear audio queue for speaker=%s", speaker_identity)

    def _recipients(self, speaker: rtc.RemoteParticipant) -> list[rtc.RemoteParticipant]:
        """Return other human participants in this room.

        Parameters
        ----------
        speaker : rtc.RemoteParticipant
            The participant currently speaking.

        Returns
        -------
        list[rtc.RemoteParticipant]
            List of eligible recipient participants.
        """
        return [
            participant
            for participant in self.room.remote_participants.values()
            if participant.identity != speaker.identity
            and not participant.identity.startswith("translator-")
        ]

    @staticmethod
    def _participant_language(
        participant: rtc.RemoteParticipant, attribute: str, default: str
    ) -> str:
        """Read a language attribute defensively from LiveKit participant metadata.

        Parameters
        ----------
        participant : rtc.RemoteParticipant
            The participant to inspect.
        attribute : str
            The attribute key to read.
        default : str
            The default fallback language code.

        Returns
        -------
        str
            The resolved language code.
        """
        attributes = participant.attributes or {}
        language = attributes.get(attribute, default)
        return str(language).strip().lower() or default

    @staticmethod
    def _voice_for_language(language: str) -> str | None:
        """Resolve a language code to a configured Piper voice key.

        Parameters
        ----------
        language : str
            The target language code.

        Returns
        -------
        str | None
            The matching voice key, or None if not found.
        """
        voices = get_settings().PIPER_VOICES
        if language in voices:
            return language
        candidates = [key for key in voices if key.split("_", maxsplit=1)[0] == language]
        if not candidates:
            return None
        return next((key for key in candidates if key.endswith("_female")), candidates[0])

    async def _handle_utterance(
        self, speaker: rtc.RemoteParticipant, frames: list[rtc.AudioFrame]
    ) -> None:
        """Transcribe one VAD-delimited utterance and deliver translations.

        Parameters
        ----------
        speaker : rtc.RemoteParticipant
            The participant who spoke.
        frames : list[rtc.AudioFrame]
            The raw audio frames comprising the utterance.
        """
        audio = frames_to_float32(frames)
        if audio.size < int(get_settings().WHISPER_SAMPLE_RATE * 0.18):
            return

        settings = get_settings()
        source_language = self._participant_language(speaker, "speak_language", "en")
        speaker_name = speaker.name or speaker.identity
        try:
            async with self._transcription_semaphore:
                transcript = await asyncio.to_thread(transcribe_pcm, audio, source_language)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("STT failed for participant=%s", speaker.identity)
            return

        transcript = transcript.strip()
        if not transcript:
            logger.debug("No speech recognized for participant=%s", speaker.identity)
            return

        recipients = self._recipients(speaker)
        if not recipients:
            logger.debug("No recipients for room=%s", self.room.name)
            return

        logger.info(
            "Transcribed utterance participant=%s language=%s chars=%d",
            speaker.identity,
            source_language,
            len(transcript),
        )

        recipients_by_language: dict[str, list[rtc.RemoteParticipant]] = {}
        for recipient in recipients:
            target = self._participant_language(recipient, "hear_language", "en")
            recipients_by_language.setdefault(target, []).append(recipient)

        for target_language, language_recipients in recipients_by_language.items():
            try:
                translated = (
                    await asyncio.to_thread(
                        translate_text, transcript, source_language, target_language
                    )
                    if settings.TRANSLATION_ENABLED
                    else transcript
                ).strip()
                if not translated:
                    translated = transcript
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Translation failed for %s -> %s", source_language, target_language
                )
                translated = transcript

            for recipient in language_recipients:
                await self._publish_caption(
                    recipient=recipient,
                    speaker=speaker,
                    speaker_name=speaker_name,
                    source_language=source_language,
                    target_language=target_language,
                    source_text=transcript,
                    translated_text=translated,
                )
                if not settings.TTS_ENABLED or not translated:
                    continue
                voice_key = self._voice_for_language(target_language)
                if voice_key is None:
                    logger.warning(
                        "No Piper voice configured for target language=%s; captions only",
                        target_language,
                    )
                    continue
                task = asyncio.create_task(
                    self._speak_to_recipient(
                        speaker=speaker,
                        recipient=recipient,
                        target_language=target_language,
                        voice_key=voice_key,
                        text=translated,
                    ),
                    name=f"tts-{recipient.identity}-{target_language}",
                )
                tasks = self._playback_tasks.setdefault(recipient.identity, set())
                tasks.add(task)
                task.add_done_callback(
                    lambda done, identity=recipient.identity: self._discard_playback(identity, done)
                )

    def _discard_playback(self, identity: str, task: asyncio.Task[None]) -> None:
        """Remove a finished playback task from the participant's task set.

        Parameters
        ----------
        identity : str
            The participant identity.
        task : asyncio.Task[None]
            The completed task.
        """
        tasks = self._playback_tasks.get(identity)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                self._playback_tasks.pop(identity, None)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error("Playback task failed for participant=%s: %s", identity, error)

    async def _publish_caption(
        self,
        *,
        recipient: rtc.RemoteParticipant,
        speaker: rtc.RemoteParticipant,
        speaker_name: str,
        source_language: str,
        target_language: str,
        source_text: str,
        translated_text: str,
    ) -> None:
        """Publish a caption data packet only to the intended recipient.

        Parameters
        ----------
        recipient : rtc.RemoteParticipant
            The intended recipient of the caption.
        speaker : rtc.RemoteParticipant
            The participant who spoke.
        speaker_name : str
            The display name of the speaker.
        source_language : str
            The source language code.
        target_language : str
            The target language code.
        source_text : str
            The original transcribed text.
        translated_text : str
            The translated text.
        """
        payload = {
            "type": "caption",
            "id": f"{speaker.identity}-{time.time_ns()}",
            "timestamp": time.time(),
            "speaker_identity": speaker.identity,
            "speaker_name": speaker_name,
            "source_language": source_language,
            "target_language": target_language,
            "source_text": source_text,
            "translated_text": translated_text,
        }
        try:
            await self.room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                reliable=True,
                destination_identities=[recipient.identity],
                topic="captions",
            )
        except Exception:
            logger.exception("Could not publish caption to participant=%s", recipient.identity)

    async def _get_or_create_output(
        self,
        speaker_identity: str,
        target_language: str,
    ) -> PublishedVoice:
        """Create and publish one persistent output track for a speaker/language pair.

        Parameters
        ----------
        speaker_identity : str
            The identity of the speaker.
        target_language : str
            The target language code.

        Returns
        -------
        PublishedVoice
            The published voice source, track, and publication wrapper.
        """
        key = (speaker_identity, target_language)
        existing = self._output_tracks.get(key)
        if existing is not None:
            return existing

        sample_rate = get_settings().PIPER_OUTPUT_SAMPLE_RATE
        source = rtc.AudioSource(sample_rate=sample_rate, num_channels=1)
        safe_identity = re.sub(r"[^a-zA-Z0-9_-]", "_", speaker_identity)[:30]
        name = f"translation-{safe_identity}-{target_language}"

        track = rtc.LocalAudioTrack.create_audio_track(name, source)

        try:
            publication = await self.room.local_participant.publish_track(
                track,
                rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_MICROPHONE,
                ),
            )
        except Exception:
            logger.exception("Failed to publish audio track for speaker=%s", speaker_identity)
            raise

        output = PublishedVoice(source=source, track=track, publication=publication)
        self._output_tracks[key] = output
        logger.info("Published translated audio track=%s", name)
        return output

    async def _speak_to_recipient(
        self,
        *,
        speaker: rtc.RemoteParticipant,
        recipient: rtc.RemoteParticipant,
        target_language: str,
        voice_key: str,
        text: str,
    ) -> None:
        """Synthesize speech and feed 10 ms PCM frames into a LiveKit audio source.

        Parameters
        ----------
        speaker : rtc.RemoteParticipant
            The participant who spoke.
        recipient : rtc.RemoteParticipant
            The recipient participant.
        target_language : str
            The target language code.
        voice_key : str
            The Piper voice key to use.
        text : str
            The text to synthesize.
        """
        output: PublishedVoice | None = None
        try:
            pcm = await asyncio.to_thread(synthesize_pcm, text, voice_key)
            if pcm.size == 0 or not np.any(pcm):
                logger.warning("TTS returned empty/silent audio for language=%s", target_language)
                return

            output = await self._get_or_create_output(speaker.identity, target_language)
            output.track.unmute()
            output_sample_rate = get_settings().PIPER_OUTPUT_SAMPLE_RATE
            frame_samples = max(1, output_sample_rate // 100)  # 10 ms

            for start in range(0, len(pcm), frame_samples):
                chunk = np.ascontiguousarray(pcm[start : start + frame_samples], dtype=np.int16)
                frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=output_sample_rate,
                    num_channels=1,
                    samples_per_channel=chunk.size,
                )
                await output.source.capture_frame(frame)

            logger.debug(
                "Queued translated audio recipient=%s samples=%d",
                recipient.identity,
                pcm.size,
            )
        except asyncio.CancelledError:
            if output is not None:
                try:
                    await clear_audio_queue(output.source)
                except Exception:
                    logger.exception("Failed to clear audio queue during cancellation")
            raise
        except Exception:
            logger.exception(
                "TTS/playback failed recipient=%s language=%s",
                recipient.identity,
                target_language,
            )

    async def close(self) -> None:
        """Stop readers, utterance tasks, and playback tasks during shutdown."""
        if self._closed:
            return
        self._closed = True
        tasks: list[asyncio.Task[Any]] = [
            *self._track_tasks.values(),
            *self._utterance_tasks,
            *(task for group in self._playback_tasks.values() for task in group),
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for output in self._output_tracks.values():
            try:
                await clear_audio_queue(output.source)
            except Exception:
                logger.exception("Failed to clear audio queue during shutdown")
        self._track_tasks.clear()
        self._utterance_tasks.clear()
        self._playback_tasks.clear()
