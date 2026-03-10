"""Application configuration."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    data_dir: Path = Path("./data/sample_dataset")
    cache_dir: Path = Path("./data/cache")
    max_chars_per_patient: int = 18_000
    top_k_search_results: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
settings.cache_dir.mkdir(parents=True, exist_ok=True)
