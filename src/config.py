from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Daily
    daily_api_key: str = ""
    daily_room_url: str = ""
    daily_api_url: str = "https://api.daily.co/v1"

    # Deepgram
    deepgram_api_key: str = ""

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/medical_triage"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"


settings = Settings()
