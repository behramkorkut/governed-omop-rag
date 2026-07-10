"""Configuration typée de l'application (pydantic-settings).

Source unique de configuration. Toutes les valeurs sont surchargées par des
variables d'environnement préfixées ``GOR_`` (ou un fichier ``.env``).
Voir ``.env.example`` pour la liste documentée.

Exemple::

    from governed_omop_rag.config import get_settings

    settings = get_settings()
    print(settings.qdrant_url)
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    """Environnement d'exécution."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class VectorBackend(StrEnum):
    """Implémentation de base vectorielle derrière l'interface VectorStore."""

    QDRANT = "qdrant"
    # Extensible plus tard sans toucher au reste du code (moindre couplage).
    MEMORY = "memory"


class Settings(BaseSettings):
    """Paramètres applicatifs chargés depuis l'environnement / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="GOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Général ---
    env: AppEnv = AppEnv.DEV
    log_level: str = "INFO"

    # --- LLM des agents (Phase 3) ---
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5"

    # --- Base vectorielle (Phase 1-2) ---
    vector_backend: VectorBackend = VectorBackend.QDRANT
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "ohdsi_concepts"

    # --- Embeddings biomédicaux locaux (Phase 1) ---
    embedding_model: str = "FremyCompany/BioLORD-2023"
    embedding_device: str = "cpu"

    # --- Corpus médaillon (couche data) ---
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/gor.duckdb")

    # --- Garde-fous / gouvernance ---
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=100)

    @property
    def bronze_dir(self) -> Path:
        """Répertoire des données brutes (couche Bronze)."""
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        """Répertoire des concepts filtrés/normalisés (couche Silver)."""
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        """Répertoire des documents embedding-ready (couche Gold)."""
        return self.data_dir / "gold"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retourne l'instance singleton des paramètres (mémoïsée)."""
    return Settings()
