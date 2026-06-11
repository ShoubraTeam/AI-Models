# -----------------------------------------------------
# General Configurations
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
    "device"       : "auto"
}

RETINA_DETECTOR_CFG = {
    "max_size": 512,
    "device"  : "auto"
}

# job desc enhancer
JOB_DESCRIPTION_ENHANCEMENT_MODELS = {
    "tools_detector"   : "llama-3.1-8b-instant",
    "tools_recommender": "llama-3.3-70b-versatile",
    "job_desc_enhancer": "llama-3.1-8b-instant"
}

JOB_DESCRIPTION_ENHANCEMENT_COLLECTION_V1 = "job_desc_enhancement_collection_v1"
JOB_DESCRIPTION_N_JOBS_TO_RETRIEVE = 10
JOB_DESCRIPTION_RETRIEVAL_ALPHA    = 0.7
JOB_DESCRIPTION_RAG_EMBEDDER = {
    "model_name"    : "BAAI/bge-base-en-v1.5",
    "model_kwargs"  : {"device" : "cuda"},
    "encode_kwargs" : {"batch_size" : 128}
}

JOB_DESCRIPTION_RAG_RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


# proposal rejection reasons
# Models
PROVIDER_GROQ         = "groq"
GROQ_LLAMA_8b         = "groq:llama-3.1-8b-instant"
GROQ_LLAMA_70b        = "groq:llama-3.3-70b-versatile"
GROQ_QWEN_32b         = "groq:qwen/qwen3-32b"
GROQ_GPT_120b         = "groq:openai/gpt-oss-120b"
GROQ_GPT_20b          = "groq:openai/gpt-oss-20b"

PROVIDER_GOOGLE_GENAI = "google_genai"
GEMINI_FLASH_LITE     = "google_genai:gemini-2.5-flash-lite"
GEMINI_FLASH          = "google_genai:gemini-2.5-flash"



# Models CFG
DEFAULT_MODELS_CFG = {
    "job_tools_extractor" : {
        "temperature" : 0.0,
        "max_tokens"  : 1024
    },

    "proposal_tools_analyzer" : {
        "temperature" : 0.0,
        "max_tokens"  : 1024
    },

    "job_key_points_extractor" : {
        "temperature": 0.0,
        "max_tokens" : 1024,
    },

    "job_understanding_evaluator" : {
        "temperature": 0.0,
        "max_tokens" : 1024
    },


    "job_requirements_extractor" : {
        "temperature": 0.0,
        "max_tokens" : 2048
    },

    "job_requirements_matcher" : {
        "temperature": 0.0,
        "max_tokens" : 2048
    },
    "experience_evidence_agent" : {
        "temperature": 0.0,
        "max_tokens" : 1024
    },
    
    "language_clarity_evaluator": {
        "temperature": 0.0,
        "max_tokens" : 1024
    },
}

# --------------------------------------------------- PRR Scoring ----------------------------------------------------

# Tools Alignment Scoring
NECESSITY_LEVEL_WEIGHTS = {
    "mandatory"  : 1,
    "forbidden"  : -1,
    "recommended": 0.7,
    "optional"   : 0.5
}


WITH_CONFIDENCE_TOOL_WEIGHT = 1
GENERIC_TOOL_WEIGHT         = 0.5

# thresholds

TA_TOOL_ALIGNMENT_THRESHOLD = 0.5
JD_JOB_UNDERSTANDING_THRESHOLD = 0.5
RQ_REQUIREMENT_COVERAGE_THRESHOLD = 0.5
LANGUAGE_CLARITY_THRESHOLD = 0.5 
EXPERIENCE_EVIDENCE_THRESHOLD = 0.5
