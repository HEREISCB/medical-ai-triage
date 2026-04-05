from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Deepgram (STT + TTS)
    deepgram_api_key: str = ""

    # Groq (LLM)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Database (optional for MVP)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/medical_triage"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    ws_port: int = 8765


settings = Settings()
