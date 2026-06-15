"""
Download and load ML models before the API accepts traffic.

Run directly: python3 -m app.preload
Called from entrypoint.sh before uvicorn / Ray Serve starts.
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preload_models() -> None:
    """Load Whisper and diarization models into memory (downloads on first run)."""
    from app.pipeline import HF_TOKEN, load_whisper_model, load_diarize_pipeline

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

    logger.info("All configured models preloaded successfully")


def main() -> None:
    try:
        preload_models()
    except Exception as exc:
        logger.error(f"Model preload failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
