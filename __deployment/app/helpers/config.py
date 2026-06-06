# -----------------------------------------------------
# General Configurations
# -----------------------------------------------------


from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import torch


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
    WEAVIATE_URL    : str
    WEAVIATE_API_KEY: str
   

def get_settings() -> Settings:
    return Settings()



# routes
ROUTE_MAIN_ROUTE = "/ai/api"



# ------------------------ Agents / Clients CFG -------------------------
# identity recognition
ARCFACE_CFG = {
    "n_classes"    : 786,
    "embedding_dim": 512,
    "margin"       : 0.5,
    "device"       : torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

RETINA_DETECTOR_CFG = {
    "max_size": 512,
    "device"  :torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# job desc enhancer
JOB_DESCRIPTION_ENHANCEMENT_MODELS = {
    "tools_detector"   : "llama-3.1-8b-instant",
    "tools_recommender": "llama-3.3-70b-versatile",
    "job_desc_enhancer": "llama-3.1-8b-instant"
}

JOB_DESCRIPTION_ENHANCEMENT_COLLECTION_V1 = "job_desc_enhancement_collection_v1"
JOB_DESCRIPTION_N_JOBS_TO_RETREIVE = 10
JOB_DESCRIPTION_RETREIVAL_ALPHA    = 0.7
JOB_DESCRIPTION_RAG_EMBEDDER = {
    "model_name"    : "BAAI/bge-base-en-v1.5",
    "model_kwargs"  : {"device" : "cuda"},
    "encode_kwargs" : {"batch_size" : 128}
}

JOB_DESCRIPTION_RAG_RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'