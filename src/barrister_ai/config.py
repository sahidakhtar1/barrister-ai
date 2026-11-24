"""Centralized configuration and directory paths for Barrister_AI."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

PROJECT_NAME = "Barrister_AI"
VERSION = "1.0.0"
ENCODING = "utf-8"
DATASET_SIZE = 500

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_BASE = DATA_DIR / "knowledge_base"
ACTS_DATA_DIR = DATA_DIR / "acts"
PRECEDENTS_DIR = DATA_DIR / "precedents"
EXPORTS_DIR = DATA_DIR / "exports"
QDRANT_STORAGE_DIR = DATA_DIR / "qdrant_storage"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL connection settings loaded from environment variables."""

    user: str = field(default_factory=lambda: os.getenv("DB_USER", "myuser"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", "mypassword"))
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: str = field(default_factory=lambda: os.getenv("DB_PORT", "5431"))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "acts"))

    @property
    def url(self) -> str:
        """Return the SQLAlchemy PostgreSQL connection URL."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass(frozen=True)
class LLMConfig:
    """Local or OpenAI-compatible LLM inference settings."""

    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "ai/gemma4:E2B"))
    base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:12434/engines/v1")
    )
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "docker"))


@dataclass(frozen=True)
class QdrantConfig:
    """Qdrant vector store and embedding model configuration."""

    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection_name: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "acts")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )


@dataclass(frozen=True)
class Settings:
    """Top-level application settings container."""

    project_name: str = PROJECT_NAME
    version: str = VERSION
    encoding: str = ENCODING
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    knowledge_base_dir: Path = KNOWLEDGE_BASE
    acts_data_dir: Path = ACTS_DATA_DIR
    precedents_dir: Path = PRECEDENTS_DIR
    exports_dir: Path = EXPORTS_DIR
    qdrant_storage_dir: Path = QDRANT_STORAGE_DIR
    reports_dir: Path = REPORTS_DIR
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)


def get_settings() -> Settings:
    """Create a Settings snapshot from the current environment."""
    return Settings()
