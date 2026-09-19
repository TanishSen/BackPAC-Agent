"""Voice services (STT + TTS) and the Pipecat pipeline for BackPAC.

Ported from the proven ajimganj concierge pipeline: Azure STT with
noise-suppression + language auto-detect/lock/reconnect, ElevenLabs
eleven_v3 TTS, Silero VAD, barge-in turn strategies, a TTS text filter
and an audio jitter buffer. Trimmed for BackPAC and given an STT_MODE
switch (azure | device). See AGENT_GUIDE.md."""

import os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.azure.stt import AzureSTTService
from loguru import logger
import asyncio
import azure.cognitiveservices.speech as speechsdk
import aiohttp

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

from .elevenlabs_v3_tts import ElevenLabsV3TTSService
from .tts_text_filter import TTSTextFilter
from .jitter_buffer import TTSAudioJitterBuffer

from pipecat.turns.user_turn_strategies import UserTurnStrategies, TranscriptionUserTurnStartStrategy, BaseUserTurnStartStrategy
from pipecat.turns.types import ProcessFrameResult
from pipecat.frames.frames import Frame


class AzureSTTWithNoiseSuppression(AzureSTTService):
    """Azure STT Service with noise suppression enabled."""

    def __init__(self, api_key: str, region: str, auto_detect_source_language_config: dict = None, **kwargs):
        super().__init__(api_key=api_key, region=region, **kwargs)

        # Enable aggressive noise suppression with AEC and AGC
        self._speech_config.set_property(
            speechsdk.PropertyId.AudioConfig_AudioProcessingOptions,
            '{"AEC":true,"AGC":true,"NS":true,"NsMode":"Aggressive"}'
        )

        # Set initial silence timeout
        self._speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "5000"
        )

        # Enable dictation mode for better recognition of punctuation, numbers, and formatting
        self._speech_config.enable_dictation()

        # Set output format to detailed for richer recognition results
        self._speech_config.output_format = speechsdk.OutputFormat.Detailed

        # Enable profanity masking to raw (unmasked) for accurate transcription
        self._speech_config.set_profanity(speechsdk.ProfanityOption.Raw)

        self.auto_detect_config = None
        if auto_detect_source_language_config and "candidate_languages" in auto_detect_source_language_config:
            candidate_langs = auto_detect_source_language_config["candidate_languages"]
            logger.info(f"Configuring Azure STT with candidate languages for auto-detect: {candidate_langs}")
            self.auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=candidate_langs
            )
            # AtStart: detect language from the first utterance only (secondary guard).
            # The primary guard is _switch_to_single_language() which kills auto-detect
            # entirely after the first recognition and restarts in single-language mode.
            self._speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                value='AtStart'
            )

        # Language lock: set to the BCP-47 code (e.g. "en-IN") after the very first
        # recognized utterance. Once set it never changes, even across reconnects.
        self._locked_language: str | None = None

        logger.info(
            "Azure STT initialized with noise suppression and dictation mode enabled")

    async def _switch_to_single_language(self, language_code: str):
        """Transition from auto-detect mode to a fixed single-language mode."""
        logger.info(f"Locking Azure STT language to: {language_code}")
        self._locked_language = language_code
        self.auto_detect_config = None
        self._speech_config.speech_recognition_language = language_code
        await self._reconnect_stt()

    async def _connect(self):
        """Initialize the Azure speech recognizer and begin continuous recognition with optional auto-detect."""
        if self._audio_stream:
            return

        try:
            stream_format = speechsdk.audio.AudioStreamFormat(samples_per_second=self.sample_rate, channels=1)
            self._audio_stream = speechsdk.audio.PushAudioInputStream(stream_format)

            audio_config = speechsdk.audio.AudioConfig(stream=self._audio_stream)

            if self.auto_detect_config:
                logger.info("Starting Azure SpeechRecognizer with AutoDetectSourceLanguageConfig")
                self._speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self._speech_config,
                    audio_config=audio_config,
                    auto_detect_source_language_config=self.auto_detect_config
                )
            else:
                self._speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self._speech_config,
                    audio_config=audio_config
                )
                
            self._speech_recognizer.recognizing.connect(self._on_handle_recognizing)
            self._speech_recognizer.recognized.connect(self._on_handle_recognized)
            self._speech_recognizer.canceled.connect(self._on_handle_canceled)
            self._speech_recognizer.start_continuous_recognition_async()
        except Exception as e:
            await self.push_error(
                error_msg=f"Uncaught exception during initialization: {e}", exception=e
            )

    def _on_handle_recognized(self, event):
        if event.result.reason == speechsdk.ResultReason.RecognizedSpeech and len(event.result.text) > 0:
            detected_lang = None
            if self.auto_detect_config:
                try:
                    auto_detect_result = speechsdk.AutoDetectSourceLanguageResult(event.result)
                    detected_lang = auto_detect_result.language
                    if detected_lang:
                        logger.info(f"Azure STT auto-detected language: {detected_lang}")
                        # First-time detection: lock the language and switch to single-language mode.
                        # This prevents all future utterances (including after reconnects) from being
                        # misclassified — e.g. Indian-accented English detected as Hindi or Bengali.
                        if self._locked_language is None:
                            logger.info(f"Azure STT: first utterance detected as '{detected_lang}' — locking language for session.")
                            asyncio.run_coroutine_threadsafe(
                                self._switch_to_single_language(detected_lang), self.get_event_loop()
                            )
                except Exception as e:
                    logger.debug(f"Failed to get auto-detect language: {e}")

            from pipecat.transcriptions.language import Language
            # Use locked language if already set, otherwise fall back to detected or default
            language = self._locked_language or detected_lang or self._settings.language or Language.EN_US

            from pipecat.frames.frames import TranscriptionFrame
            from pipecat.utils.time import time_now_iso8601

            frame = TranscriptionFrame(
                event.result.text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=event,
            )
            asyncio.run_coroutine_threadsafe(
                self._handle_transcription(event.result.text, True, language), self.get_event_loop()
            )
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())

    def _on_handle_recognizing(self, event):
        if event.result.reason == speechsdk.ResultReason.RecognizingSpeech and len(event.result.text) > 0:
            detected_lang = None
            if self.auto_detect_config:
                try:
                    auto_detect_result = speechsdk.AutoDetectSourceLanguageResult(event.result)
                    detected_lang = auto_detect_result.language
                except Exception as e:
                    pass
            
            from pipecat.transcriptions.language import Language
            language = detected_lang or self._settings.language or Language.EN_US
            
            from pipecat.frames.frames import InterimTranscriptionFrame
            from pipecat.utils.time import time_now_iso8601
            
            frame = InterimTranscriptionFrame(
                event.result.text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=event,
            )
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())

    def _on_handle_canceled(self, event):
        """Handle Azure STT cancellation events.

        If the cancellation is due to a buffer overflow (service inactivity
        caused the client buffer to exceed its maximum size), we silently
        reconnect instead of propagating a fatal error that would kill the
        pipeline session.
        """
        details = event.result.cancellation_details
        if details.reason == speechsdk.CancellationReason.Error and details.error_details:
            if "client buffer exceeded maximum size" in details.error_details:
                logger.warning(
                    f"Azure STT buffer overflow detected — scheduling reconnect. "
                    f"Details: {details.error_details}"
                )
                asyncio.run_coroutine_threadsafe(
                    self._reconnect_stt(), self.get_event_loop()
                )
                return  # Recoverable — don't push error, reconnect handles it
        # For any other cancellation reason, fall through to the base handler
        super()._on_handle_canceled(event)

    async def _reconnect_stt(self):
        """Tear down the current Azure recognizer and start a fresh session.

        Called automatically when a buffer-overflow cancellation is detected.
        """
        logger.info("Azure STT reconnecting after buffer overflow…")
        try:
            if self._speech_recognizer:
                try:
                    self._speech_recognizer.stop_continuous_recognition_async().get()
                except Exception as stop_err:
                    logger.debug(f"Azure STT stop error (ignorable): {stop_err}")
                self._speech_recognizer = None

            if self._audio_stream:
                try:
                    self._audio_stream.close()
                except Exception as close_err:
                    logger.debug(f"Azure STT audio stream close error (ignorable): {close_err}")
                self._audio_stream = None

            await self._connect()
            logger.info("Azure STT reconnected successfully after buffer overflow.")
        except Exception as e:
            logger.error(f"Azure STT reconnect failed: {e}")
            await self.push_error(error_msg=f"Azure STT reconnect failed: {e}", exception=e)


def _as_pipecat_language(code: str):
    """"en-IN" -> Language.EN_IN. Falls back to the raw string, which Azure
    accepts anyway, so an unlisted locale still works."""
    from pipecat.transcriptions.language import Language

    return getattr(Language, code.replace("-", "_").upper(), code)


def _opt_float(name: str) -> float | None:
    val = os.getenv(name)
    return float(val) if val else None


def create_services(
    voice_id: str = None,
    language: str = None,
    aiohttp_session: aiohttp.ClientSession = None,
):
    """Build (stt, tts).

    **Language.** `STT_LANGUAGES` is a comma-separated list of BCP-47 codes and
    defaults to `en-IN` alone, which means Azure is told the language rather
    than asked to guess it.

    Guessing is off by default because of how badly it fails. Azure decides from
    the *first* utterance it hears, and the first thing it hears is often not
    the traveller — it is the agent's own greeting coming back off a laptop
    speaker, or a door closing. Guess "hi-IN" from that and every English
    sentence for the rest of the call comes back transliterated into Devanagari
    ("आई एम जस्ट वांट टू प्लान माय ट्रिप"). The words are legible to Claude,
    which is why this is easy to miss, but the transcript on screen looks
    broken and the wrong acoustic model costs accuracy.

    Set `STT_LANGUAGES=en-IN,hi-IN,bn-IN` to opt back into auto-detect. With
    more than one code the detected language is still *locked* after the first
    utterance, so it cannot drift mid-call — an Indian-accented English speaker
    otherwise slides into being classified as Hindi halfway through.

    `STT_MODE=device` returns `stt=None`: the phone transcribes and posts text
    over the data channel, so no server STT is built at all.
    """
    if not aiohttp_session:
        raise ValueError("ElevenLabsV3TTSService requires an aiohttp_session")

    if os.getenv("STT_MODE", "azure").lower() == "device":
        logger.info("STT_MODE=device — the phone transcribes; no server STT built.")
        stt = None
    else:
        languages = [
            code.strip()
            for code in (language or os.getenv("STT_LANGUAGES", "en-IN")).split(",")
            if code.strip()
        ] or ["en-IN"]

        common = {
            "api_key": os.getenv("AZURE_SPEECH_API_KEY"),
            "region": os.getenv("AZURE_SPEECH_REGION"),
        }
        if len(languages) == 1:
            logger.info(f"Azure STT fixed to {languages[0]}")
            stt = AzureSTTWithNoiseSuppression(
                **common, language=_as_pipecat_language(languages[0])
            )
        else:
            logger.info(
                f"Azure STT auto-detecting between {languages} "
                "(locks to the first one recognised)"
            )
            stt = AzureSTTWithNoiseSuppression(
                **common,
                auto_detect_source_language_config={"candidate_languages": languages},
            )

    model_name = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
    logger.info(f"ElevenLabs TTS: model={model_name} voice={voice_id}")

    tts = ElevenLabsV3TTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=voice_id,
        aiohttp_session=aiohttp_session,
        model=model_name,
        params=ElevenLabsV3TTSService.InputParams(
            speed=_opt_float("ELEVENLABS_SPEED"),
            stability=_opt_float("ELEVENLABS_STABILITY"),
            similarity_boost=_opt_float("ELEVENLABS_SIMILARITY_BOOST"),
            style=_opt_float("ELEVENLABS_STYLE")
        )
    )
    return stt, tts


def create_pipeline(transport, aic_filter, stt, tts, unified_langgraph_processor, room_name):
    """Create the processing pipeline."""
    context = LLMContext()
    
    if aic_filter:
        logger.info("Initializing integrated AICVADAnalyzer from AICFilter")
        try:
            vad_sensitivity        = float(os.getenv("AIC_VAD_SENSITIVITY", "2.5"))
            vad_hold_duration      = float(os.getenv("AIC_VAD_HOLD_DURATION", "0.25"))
            vad_min_speech         = float(os.getenv("AIC_VAD_MIN_SPEECH_DURATION", "0.25"))
            logger.info(
                f"AIC VAD config — sensitivity={vad_sensitivity}, "
                f"hold_duration={vad_hold_duration}s, min_speech={vad_min_speech}s"
            )
            vad_analyzer = aic_filter.create_vad_analyzer(
                speech_hold_duration=vad_hold_duration,
                minimum_speech_duration=vad_min_speech,
                sensitivity=vad_sensitivity,
            )
        except Exception as e:
            logger.error(f"Failed to create AIC VAD Analyzer: {e}. Falling back to SileroVADAnalyzer.")
            vad_analyzer = SileroVADAnalyzer()
    else:
        logger.info("AICFilter not active. Falling back to SileroVADAnalyzer")
        vad_analyzer = SileroVADAnalyzer()



    class DummyUserTurnStartStrategy(BaseUserTurnStartStrategy):
        async def process_frame(self, frame: Frame) -> ProcessFrameResult:
            return ProcessFrameResult.CONTINUE

    disable_interruptions = os.getenv("DISABLE_INTERRUPTIONS", "false").lower() == "true"
    transcription_only = os.getenv("INTERRUPT_ON_TRANSCRIPTION_ONLY", "false").lower() == "true"

    if disable_interruptions:
        logger.info("Interruptions disabled: Using DummyUserTurnStartStrategy")
        user_turn_strategies = UserTurnStrategies(
            start=[DummyUserTurnStartStrategy()]
        )
    elif transcription_only:
        logger.info("Interruptions restricted to transcription only (VAD noises ignored)")
        user_turn_strategies = UserTurnStrategies(
            start=[TranscriptionUserTurnStartStrategy()]
        )
    else:
        user_turn_strategies = None

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
            user_idle_timeout=300,
            user_turn_strategies=user_turn_strategies,
        ),
    )
    from .card_dispatcher import CardDispatcherProcessor
    from .text_input import TextInputProcessor
    from .word_interceptor import WordInterceptor
    card_dispatcher = CardDispatcherProcessor()
    word_interceptor = WordInterceptor(transport)
    # Typed turns from the app arrive as transport-message frames; this turns
    # them into turns on the same brain the microphone feeds.
    text_input = TextInputProcessor(unified_langgraph_processor)

    # Instantiate TTS filter processors
    tts_text_filter = TTSTextFilter()
    tts_jitter_buffer = TTSAudioJitterBuffer(
        target_ms=int(os.getenv("TTS_JITTER_BUFFER_MS", "200"))
    )

    # Data flow: mic in -> [STT] -> user aggregator -> typed-turn input ->
    # the LangGraph brain -> cards to the app -> TTS text cleanup -> transcript
    # to the app -> ElevenLabs -> jitter buffer -> speaker out -> assistant
    # aggregator. In device mode `stt` is None (the phone already transcribed),
    # so that stage is simply omitted.
    #
    # The transcript is published *after* the text filter on purpose: the model
    # sometimes emits markdown ("**Haveli Courtyard**") and ElevenLabs audio
    # tags ("[warmly]"). The filter strips both before they are spoken, so
    # publishing before it would show the app asterisks and stage directions
    # that nobody ever hears.
    stages = [transport.input()]
    if stt is not None:
        stages.append(stt)
    stages += [
        aggregators.user(),
        text_input,           # typed turns in, alongside spoken ones
        unified_langgraph_processor,
        card_dispatcher,
        tts_text_filter,      # strip markdown / SSML / audio tags
        word_interceptor,     # publish the SAME cleaned text to the app
        tts,
        tts_jitter_buffer,    # smooth audio right after TTS
        transport.output(),
        aggregators.assistant(),
    ]
    pipeline = Pipeline(stages)
    return pipeline, aggregators, word_interceptor



