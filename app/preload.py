"""
Download and load ML models before the API accepts traffic.

Run directly: python3 -m app.preload
Called from FastAPI startup (and Ray Serve deployments).
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PRELOAD_ALIGN_LANGUAGES = ("en", "ar")


def preload_align_models() -> None:
    """Download alignment (wav2vec2) models for configured languages."""
    from app.pipeline import load_align_model, clear_gpu_memory

    for language_code in PRELOAD_ALIGN_LANGUAGES:
        try:
            logger.info(f"Preloading alignment model for: {language_code}")
            load_align_model(language_code)
            logger.info(f"Alignment model ready: {language_code}")
        except Exception as exc:
            logger.warning(
                f"Could not preload alignment model for {language_code}: {exc}"
            )

    clear_gpu_memory()


def preload_models() -> None:
    """Load Whisper, diarization, and alignment models."""
    from app.pipeline import HF_TOKEN, load_whisper_model, load_diarize_pipeline, clear_gpu_memory

    preload_disabled = os.getenv("PRELOAD_ON_STARTUP", "true").lower() in ("0", "false", "no")
    if preload_disabled:
        logger.info("PRELOAD_ON_STARTUP=false, skipping model preload")
        return

    whisper_model = os.getenv("PRELOAD_MODEL", "large-v3").strip()
    if whisper_model:
        logger.info(f"Preloading Whisper model: {whisper_model}")
        load_whisper_model(whisper_model)
        logger.info(f"Whisper model ready: {whisper_model}")

    preload_diarization = os.getenv("PRELOAD_DIARIZATION", "true").lower() not in ("0", "false", "no")
    if preload_diarization:
        if HF_TOKEN:
            logger.info("Preloading diarization pipeline")
            load_diarize_pipeline()
            logger.info("Diarization pipeline ready")
        else:
            logger.info("Skipping diarization preload (HF_TOKEN not set)")

    preload_align_models()

    clear_gpu_memory()
    logger.info("All configured models preloaded successfully")


def main() -> None:
    try:
        preload_models()
    except Exception as exc:
        logger.error(f"Model preload failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
