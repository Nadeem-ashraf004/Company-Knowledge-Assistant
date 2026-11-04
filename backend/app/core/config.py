
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Company Knowledge Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    APP_ENV: str = "development"

    # API
    API_PREFIX: str = "/api"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    # PostgreSQL
    DATABASE_URL: str = (
        "postgresql+psycopg://postgres:password@localhost:5432/company_knowledge"
    )

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "company_documents"

    # LLM
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-2.5-flash"
    GEMINI_API_KEY: str | None = None

    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # Authentication
    SECRET_KEY: str = Field(min_length=32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    #fail-fast configuration validation.
    @field_validator("APP_ENV")
    @classmethod
    def validate_app_env(cls, value: str) -> str:
        allowed = {"development", "testing", "production"}

        if value not in allowed:
            raise ValueError(
                f"APP_ENV must be one of {allowed}"
        )

        return value


    @model_validator(mode="after")
    def validate_production_settings(self):
        if self.APP_ENV == "production":
            if self.DEBUG:
                raise ValueError("DEBUG must be false in production.")

            if not self.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required in production.")

            if not self.QDRANT_API_KEY:
                raise ValueError("QDRANT_API_KEY is required in production.")

        return self


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

