"""
Shared ASR pipeline stage functions.

Extracts the ASR pipeline (diarize -> transcribe per chunk, or transcribe -> align -> diarize)
into reusable functions consumed by both the legacy FastAPI endpoints and the
Ray Serve deployments.
"""

import os
import gc
import math
import time
import logging
import threading
import warnings
from typing import Optional, Dict, Any, Tuple, List

# Suppress pyannote's torchcodec warning -- we decode audio via whisperx.load_audio (ffmpeg),
# not pyannote's built-in decoder, so the missing torchcodec is irrelevant.
warnings.filterwarnings("ignore", message=".*torchcodec.*")

import numpy as np
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at import time, same as before)
# ---------------------------------------------------------------------------
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16" if DEVICE == "cuda" else "2"))
HF_TOKEN = os.getenv("HF_TOKEN", None)
CACHE_DIR = os.getenv("CACHE_DIR", "/.cache")
DEFAULT_MODEL = os.getenv("PRELOAD_MODEL", "large-v3")

# Idle model eviction. Set MODEL_KEEP_ALIVE_SECONDS > 0 to unload Whisper
# models that have not been used in that many seconds. Floor of 30s on the
# sweep interval to avoid pegging a thread on tight loops.
MODEL_KEEP_ALIVE_SECONDS = int(os.getenv("MODEL_KEEP_ALIVE_SECONDS", "0"))
MODEL_EVICTION_INTERVAL_SECONDS = max(
    30, int(os.getenv("MODEL_EVICTION_INTERVAL_SECONDS", "60"))
)
SAMPLE_RATE = 16000
MIN_DIARIZED_CHUNK_SECONDS = 0.05


def get_canonical_models() -> list:
    """
    Canonical model names accepted by the underlying faster-whisper engine.

    Sourced from faster_whisper.available_models() so this list stays in sync
    with whatever version of faster-whisper is installed, instead of being
    hardcoded here.
    """
    try:
        from faster_whisper import available_models
        return list(available_models())
    except Exception:
        # Defensive fallback if the import surface ever changes upstream.
        return [
            "tiny.en", "tiny", "base.en", "base", "small.en", "small",
            "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large",
            "distil-large-v2", "distil-medium.en", "distil-small.en",
            "distil-large-v3", "distil-large-v3.5", "large-v3-turbo", "turbo",
        ]


# OpenAI-style aliases → canonical faster-whisper names. These are kept for
# backwards compatibility on the request path; new clients should use the
# canonical names returned by /v1/models.
_MODEL_ALIASES = {
    "whisper-1": os.getenv("OPENAI_WHISPER1_MODEL", DEFAULT_MODEL),
    "whisper-large-v3": "large-v3",
    "whisper-large-v2": "large-v2",
    "whisper-medium": "medium",
    "whisper-small": "small",
    "whisper-base": "base",
    "whisper-tiny": "tiny",
}


def resolve_model_name(model: str) -> str:
    """
    Resolve a user-supplied model identifier to a canonical faster-whisper name.

    Accepts canonical names (tiny, large-v3, distil-medium.en, ...) as-is and
    maps OpenAI-style aliases (whisper-tiny, whisper-large-v3, ...) to their
    canonical equivalents. Unknown values are returned unchanged so the engine
    can produce its own validation error.
    """
    if not model:
        return DEFAULT_MODEL
    canonical = set(get_canonical_models())
    if model in canonical:
        return model
    if model in _MODEL_ALIASES:
        return _MODEL_ALIASES[model]
    if model.startswith("whisper-"):
        stripped = model[len("whisper-"):]
        if stripped in canonical:
            return stripped
    return model


_model_load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Model caches
# ---------------------------------------------------------------------------
_whisper_models: Dict[str, Any] = {}
_whisper_models_last_used: Dict[str, float] = {}
_align_models: Dict[str, Tuple[Any, Any]] = {}
_diarize_pipeline: Optional[DiarizationPipeline] = None

_eviction_thread_lock = threading.Lock()
_eviction_thread_started = False


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def clear_gpu_memory():
    """Clear GPU memory cache to prevent VRAM buildup."""
    if DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")


def _extract_audio_slice(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    """Extract a time-bounded slice from a 16 kHz mono audio array."""
    start_sample = max(0, int(start * SAMPLE_RATE))
    end_sample = min(len(audio), int(end * SAMPLE_RATE))
    if end_sample <= start_sample:
        return np.array([], dtype=audio.dtype)
    return audio[start_sample:end_sample]


def _offset_segment_times(segments: List[dict], offset: float, speaker: str) -> List[dict]:
    """Shift segment timestamps to absolute time and attach the speaker label."""
    shifted_segments = []
    for segment in segments:
        shifted = dict(segment)
        shifted["start"] = shifted.get("start", 0.0) + offset
        shifted["end"] = shifted.get("end", 0.0) + offset
        shifted["speaker"] = speaker
        words = shifted.get("words")
        if words:
            shifted["words"] = [
                {
                    **word,
                    "start": word.get("start", 0.0) + offset,
                    "end": word.get("end", 0.0) + offset,
                }
                for word in words
            ]
        shifted_segments.append(shifted)
    return shifted_segments


def _run_diarization(
    audio: np.ndarray,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_speaker_embeddings: bool = False,
) -> Tuple[Any, Optional[dict]]:
    """Run pyannote diarization and return speaker segments."""
    logger.info("Starting speaker diarization...")
    diarize_model = load_diarize_pipeline()

    diarize_params: Dict[str, Any] = {}
    if num_speakers is not None:
        diarize_params["num_speakers"] = num_speakers
        logger.info(f"Diarization with exact speaker count: {num_speakers}")
    else:
        if min_speakers is not None:
            diarize_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_params["max_speakers"] = max_speakers
        logger.info(f"Diarization with speaker range: {min_speakers}-{max_speakers}")

    if return_speaker_embeddings:
        diarize_params["return_embeddings"] = True
        logger.info("Speaker embeddings will be returned")

    diarize_output = diarize_model(audio, **diarize_params)

    speaker_embeddings = None
    if return_speaker_embeddings and isinstance(diarize_output, tuple):
        diarize_segments, speaker_embeddings = diarize_output
        logger.info(f"Received speaker embeddings for {len(speaker_embeddings)} speakers")
    else:
        diarize_segments = diarize_output

    if hasattr(diarize_segments, "exclusive_speaker_diarization"):
        diarize_segments = diarize_segments.exclusive_speaker_diarization
        logger.info("Using exclusive speaker diarization for better timestamp reconciliation")

    clear_gpu_memory()
    return diarize_segments, speaker_embeddings


def _transcribe_diarized_chunks(
    audio: np.ndarray,
    diarize_df: Any,
    model_name: str = DEFAULT_MODEL,
    task: str = "transcribe",
    word_timestamps: bool = True,
) -> dict:
    """
    Transcribe each diarization segment separately with language auto-detection.

    Diarization runs first; each speaker chunk is sent to Whisper independently
    so different speakers can be transcribed in different languages.
    """
    if diarize_df is None or len(diarize_df) == 0:
        return {"segments": [], "language": "en", "word_segments": []}

    all_segments: List[dict] = []
    all_word_segments: List[dict] = []
    detected_languages: List[str] = []

    diarize_rows = diarize_df.sort_values("start")
    logger.info(f"Transcribing {len(diarize_rows)} diarized speaker chunks...")

    for chunk_index, (_, row) in enumerate(diarize_rows.iterrows(), start=1):
        chunk_start = float(row["start"])
        chunk_end = float(row["end"])
        speaker = row["speaker"]
        chunk_audio = _extract_audio_slice(audio, chunk_start, chunk_end)
        chunk_duration = len(chunk_audio) / SAMPLE_RATE

        if chunk_duration < MIN_DIARIZED_CHUNK_SECONDS:
            logger.debug(
                f"Skipping diarized chunk {chunk_index} ({speaker}, "
                f"{chunk_duration:.3f}s): too short"
            )
            continue

        logger.info(
            f"Transcribing diarized chunk {chunk_index}/{len(diarize_rows)} "
            f"({speaker}, {chunk_start:.2f}s-{chunk_end:.2f}s)"
        )

        chunk_result = transcribe(
            chunk_audio,
            model_name=model_name,
            language=None,
            task=task,
            initial_prompt=None,
            hotwords=None,
        )

        chunk_language = chunk_result.get("language")
        if chunk_language:
            detected_languages.append(chunk_language)

        chunk_segments = chunk_result.get("segments", [])
        if not chunk_segments and chunk_result.get("text"):
            chunk_segments = [{
                "start": 0.0,
                "end": chunk_duration,
                "text": chunk_result["text"],
            }]

        if word_timestamps and chunk_segments:
            chunk_result = align(chunk_audio, {"segments": chunk_segments, "language": chunk_language})
            chunk_segments = chunk_result.get("segments", chunk_segments)
            chunk_word_segments = chunk_result.get("word_segments", [])
            for word_segment in chunk_word_segments:
                word_segment = dict(word_segment)
                word_segment["start"] = word_segment.get("start", 0.0) + chunk_start
                word_segment["end"] = word_segment.get("end", 0.0) + chunk_start
                word_segment["speaker"] = speaker
                if chunk_language:
                    word_segment["language"] = chunk_language
                all_word_segments.append(word_segment)

        for segment in _offset_segment_times(chunk_segments, chunk_start, speaker):
            if chunk_language:
                segment["language"] = chunk_language
            all_segments.append(segment)

    all_segments.sort(key=lambda segment: segment.get("start", 0.0))
    all_word_segments.sort(key=lambda segment: segment.get("start", 0.0))

    unique_languages = list(dict.fromkeys(detected_languages))
    if len(unique_languages) == 1:
        language = unique_languages[0]
    elif len(unique_languages) > 1:
        language = "multilingual"
    else:
        language = "en"

    logger.info(
        f"Diarize-first transcription complete: {len(all_segments)} segments, "
        f"language={language}"
    )
    return {
        "segments": all_segments,
        "language": language,
        "word_segments": all_word_segments,
    }


def _run_diarize_first_pipeline(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    task: str = "transcribe",
    word_timestamps: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_speaker_embeddings: bool = False,
) -> Tuple[dict, Optional[dict]]:
    """Run diarization first, then transcribe each speaker chunk separately."""
    logger.info("Starting diarize-first pipeline...")
    diarize_df, speaker_embeddings = _run_diarization(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_speaker_embeddings=return_speaker_embeddings,
    )
    result = _transcribe_diarized_chunks(
        audio,
        diarize_df,
        model_name=model_name,
        task=task,
        word_timestamps=word_timestamps,
    )
    return result, speaker_embeddings


# ---------------------------------------------------------------------------
# Stage 0 -- model loading
# ---------------------------------------------------------------------------
def load_whisper_model(model_name: str):
    """Load WhisperX model with caching (thread-safe)."""
    if model_name not in _whisper_models:
        with _model_load_lock:
            if model_name not in _whisper_models:
                logger.info(f"Loading WhisperX model: {model_name}")
                model = whisperx.load_model(
                    model_name,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                    download_root=CACHE_DIR,
                )
                _whisper_models[model_name] = model
                logger.info(f"Model {model_name} loaded successfully")
                # Pre-register the eviction counter time series for this model
                # so the row appears in /metrics with value 0 from the moment
                # the model is loaded, instead of only after the first eviction.
                try:
                    from app import metrics as prom_metrics
                    prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=model_name)
                except Exception:
                    pass
    _whisper_models_last_used[model_name] = time.time()
    _ensure_eviction_thread()
    return _whisper_models[model_name]


def _ensure_eviction_thread():
    """Lazily start the idle-model eviction daemon (no-op if disabled)."""
    global _eviction_thread_started
    if MODEL_KEEP_ALIVE_SECONDS <= 0 or _eviction_thread_started:
        return
    with _eviction_thread_lock:
        if _eviction_thread_started:
            return
        t = threading.Thread(
            target=_eviction_loop, daemon=True, name="model-evictor"
        )
        t.start()
        _eviction_thread_started = True
        logger.info(
            f"Idle model eviction enabled: unload after "
            f"{MODEL_KEEP_ALIVE_SECONDS}s idle, sweep every "
            f"{MODEL_EVICTION_INTERVAL_SECONDS}s"
        )


def _eviction_loop():
    while True:
        time.sleep(MODEL_EVICTION_INTERVAL_SECONDS)
        if MODEL_KEEP_ALIVE_SECONDS <= 0:
            continue
        now = time.time()
        candidates = [
            name for name, last in list(_whisper_models_last_used.items())
            if now - last > MODEL_KEEP_ALIVE_SECONDS and name in _whisper_models
        ]
        evicted_any = False
        for name in candidates:
            with _model_load_lock:
                last = _whisper_models_last_used.get(name, 0)
                if name in _whisper_models and now - last > MODEL_KEEP_ALIVE_SECONDS:
                    logger.info(f"Evicting idle model {name}")
                    del _whisper_models[name]
                    _whisper_models_last_used.pop(name, None)
                    evicted_any = True
                    try:
                        from app import metrics as prom_metrics
                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=name).inc()
                    except Exception:
                        pass
        if evicted_any:
            clear_gpu_memory()


def load_align_model(language_code: str):
    """Load alignment model with per-language caching (thread-safe)."""
    if language_code not in _align_models:
        with _model_load_lock:
            if language_code not in _align_models:
                logger.info(f"Loading alignment model for language: {language_code}")
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=DEVICE,
                    model_dir=CACHE_DIR,
                )
                _align_models[language_code] = (model_a, metadata)
                logger.info(f"Alignment model for {language_code} loaded")
    return _align_models[language_code]


def load_diarize_pipeline() -> DiarizationPipeline:
    """Load diarization pipeline (singleton, thread-safe)."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        with _model_load_lock:
            if _diarize_pipeline is None:
                logger.info("Loading diarization pipeline: pyannote/speaker-diarization-community-1")
                _diarize_pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-community-1",
                    use_auth_token=HF_TOKEN,
                    device=torch.device(DEVICE),
                )
                logger.info("Diarization pipeline loaded")
    return _diarize_pipeline


# ---------------------------------------------------------------------------
# Stage 1 -- Transcription
# ---------------------------------------------------------------------------
def transcribe(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    task: str = "transcribe",
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
) -> dict:
    """Run WhisperX transcription and return raw result dict."""
    whisper_model = load_whisper_model(model_name)

    # Set per-request options on the model's transcription options.
    # The model is cached/shared, so we must reset after transcription.
    if hotwords is not None:
        whisper_model.options.hotwords = hotwords
    if initial_prompt is not None:
        whisper_model.options.initial_prompt = initial_prompt

    transcribe_options: Dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "language": language,
        "task": task,
    }

    logger.info("Starting transcription...")
    try:
        result = whisper_model.transcribe(audio, **transcribe_options)
    finally:
        if hotwords is not None:
            whisper_model.options.hotwords = None
        if initial_prompt is not None:
            whisper_model.options.initial_prompt = None

    detected_language = result.get("language", language or "en")
    logger.info(f"Transcription complete. Detected language: {detected_language}")

    clear_gpu_memory()
    return result


# ---------------------------------------------------------------------------
# Stage 2 -- Alignment
# ---------------------------------------------------------------------------
def align(audio: np.ndarray, result: dict) -> dict:
    """Run Wav2Vec2 alignment to get word-level timestamps."""
    detected_language = result.get("language", "en")
    logger.info("Aligning timestamps...")
    try:
        model_a, metadata = load_align_model(detected_language)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )
        logger.info("Timestamp alignment complete")
        clear_gpu_memory()
    except Exception as e:
        logger.warning(f"Timestamp alignment failed: {e}, continuing without word-level timestamps")
    return result


# ---------------------------------------------------------------------------
# Stage 3 -- Diarization
# ---------------------------------------------------------------------------
def diarize(
    audio: np.ndarray,
    result: dict,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_speaker_embeddings: bool = False,
) -> Tuple[dict, Optional[dict]]:
    """
    Run pyannote speaker diarization and assign speakers to segments.

    Returns (result_with_speakers, speaker_embeddings_or_None).
    """
    if not HF_TOKEN:
        logger.warning("Speaker diarization requested but HF_TOKEN not set")
        return result, None

    logger.info("Starting speaker diarization...")
    speaker_embeddings = None
    try:
        diarize_segments, speaker_embeddings = _run_diarization(
            audio,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
        )

        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info("Speaker diarization complete")
    except Exception as e:
        logger.warning(f"Speaker diarization failed: {e}, continuing without diarization")

    return result, speaker_embeddings


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------
def sanitize_float_values(obj):
    """Recursively sanitize float values for JSON compliance (NaN/Inf -> None)."""
    if isinstance(obj, dict):
        return {key: sanitize_float_values(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_float_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_float_values(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return obj


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------
def run_pipeline(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    task: str = "transcribe",
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
    word_timestamps: bool = True,
    should_diarize: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_speaker_embeddings: bool = False,
) -> Tuple[dict, Optional[dict]]:
    """
    Run the ASR pipeline.

    When diarization is enabled, pyannote runs first and each speaker chunk is
    transcribed separately with language auto-detection. Otherwise the pipeline
    transcribes the full audio, optionally aligns, then optionally diarizes.
    """
    speaker_embeddings = None
    if should_diarize:
        if not HF_TOKEN:
            logger.warning("Speaker diarization requested but HF_TOKEN not set")
        else:
            if initial_prompt is not None or hotwords is not None:
                logger.info("initial_prompt and hotwords are ignored in diarize-first mode")
            try:
                return _run_diarize_first_pipeline(
                    audio,
                    model_name=model_name,
                    task=task,
                    word_timestamps=word_timestamps,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    return_speaker_embeddings=return_speaker_embeddings,
                )
            except Exception as e:
                logger.warning(
                    f"Diarize-first pipeline failed: {e}, falling back to transcribe-first"
                )

    result = transcribe(
        audio,
        model_name=model_name,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )

    if word_timestamps:
        result = align(audio, result)

    if should_diarize:
        result, speaker_embeddings = diarize(
            audio,
            result,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
        )

    return result, speaker_embeddings
