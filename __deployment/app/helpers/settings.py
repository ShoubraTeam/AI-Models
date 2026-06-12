# -----------------------------------------------------
# General Configurations & System Settings
# -----------------------------------------------------


from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path



# printing utils
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"


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
    APP_NAME   : str
    APP_VERSION: str

    # app config
    RESULTS_PATH                         : str
    TRAINED_MODELS_PATH                  : str
    JOB_DESCRIPTION_ENHANCEMENT_DATA_PATH: str

    # secrets
    GROQ_API_KEY    : str
    GOOGLE_API_KEY  : str
    WEAVIATE_URL    : str
    WEAVIATE_API_KEY: str

   

def get_settings() -> Settings:
    return Settings()



# routes
ROUTE_MAIN_ROUTE = "/ai/api"




# Models CFG
PROVIDER_GROQ         = "groq"
GROQ_LLAMA_8b         = "llama-3.1-8b-instant"
GROQ_LLAMA_70b        = "llama-3.3-70b-versatile"
GROQ_QWEN_32b         = "qwen/qwen3-32b"
GROQ_GPT_120b         = "openai/gpt-oss-120b"
GROQ_GPT_20b          = "openai/gpt-oss-20b"

PROVIDER_GOOGLE_GENAI = "google_genai"
GEMINI_FLASH_LITE     = "gemini-2.5-flash-lite"
GEMINI_FLASH          = "gemini-2.5-flash"



