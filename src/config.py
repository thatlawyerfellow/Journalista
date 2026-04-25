from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    openai_model: str
    openai_voice_model: str
    reasoning_effort: str
    max_output_tokens: int
    database_path: Path
    upload_dir: Path
    app_secret_key: str
    admin_username: str
    admin_password: str
    max_sample_files: int
    recommended_min_sample_files: int
    max_text_chars_per_file: int
    max_total_context_chars: int
    max_images_per_request: int
    article_max_words: int
    article_word_tolerance_pct: int
    article_parallel_threshold_words: int
    article_parallel_max_workers: int
    article_section_target_words: int


def load_config() -> AppConfig:
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4").strip(),
        openai_voice_model=os.getenv("OPENAI_VOICE_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4")).strip(),
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium").strip().lower(),
        max_output_tokens=_int_env("OPENAI_MAX_OUTPUT_TOKENS", 30000),
        database_path=Path(os.getenv("APP_DATABASE_PATH", "data/app.db")),
        upload_dir=Path(os.getenv("APP_UPLOAD_DIR", "data/uploads")),
        app_secret_key=os.getenv("APP_SECRET_KEY", "change-this-before-production"),
        admin_username=os.getenv("ADMIN_USERNAME", "admin").strip() or "admin",
        admin_password=os.getenv("ADMIN_PASSWORD", "admin"),
        max_sample_files=_int_env("MAX_SAMPLE_FILES", 50),
        recommended_min_sample_files=_int_env("RECOMMENDED_MIN_SAMPLE_FILES", 25),
        max_text_chars_per_file=_int_env("MAX_TEXT_CHARS_PER_FILE", 120000),
        max_total_context_chars=_int_env("MAX_TOTAL_CONTEXT_CHARS", 250000),
        max_images_per_request=_int_env("MAX_IMAGES_PER_REQUEST", 8),
        article_max_words=_int_env("ARTICLE_MAX_WORDS", 10000),
        article_word_tolerance_pct=_int_env("ARTICLE_WORD_TOLERANCE_PCT", 10),
        article_parallel_threshold_words=_int_env("ARTICLE_PARALLEL_THRESHOLD_WORDS", 2500),
        article_parallel_max_workers=_int_env("ARTICLE_PARALLEL_MAX_WORKERS", 4),
        article_section_target_words=_int_env("ARTICLE_SECTION_TARGET_WORDS", 1200),
    )
