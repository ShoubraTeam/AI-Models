# -----------------------------------------------------
# General Configurations
# -----------------------------------------------------


from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
class Settings(BaseSettings):
    """
    Application Settings
    """
    model_config = SettingsConfigDict(
        env_file          = ENV_PATH,
        env_file_encoding = "utf-8"
    )

    # app info
    APP_NAME: str

    

    # app config
    RESULTS_PATH: str
    

def get_settings() -> Settings:
    return Settings()