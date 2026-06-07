# ----------------------------------------------------------------
# General Configurations
# ----------------------------------------------------------------
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
PYTHON_DETERMINISTIC = "deterministic_python"

import os

# Models CFG
DEFAULT_MODELS_CFG = {
    "visual_brand_evaluator" : {
        "temperature" : 0.2,
        "max_tokens"  : 1024
    },
    "bio_analyzer" : {
        "temperature" : 0.3,
        "max_tokens"  : 1024
    },
    "skills_analyzer" : {
        "temperature" : 0.0,
        "max_tokens"  : 1024
    },
    "super_agent" : {
        "temperature" : 0.2, 
        "max_tokens"  : 1024
    }
}

# printing utils
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"



# Evaluation CFG
EVALUATION_MODELS_MAPPING = {
    "LLAMA_8B": GROQ_LLAMA_8b, 
    "GEMINI_FLASH_LITE": GEMINI_FLASH_LITE,
    "GPT_OSS_20B": GROQ_GPT_20b,
    "QWEN_32B": GROQ_QWEN_32b,
    "GEMINI_FLASH": GEMINI_FLASH,      
    "LLAMA_70B": GROQ_LLAMA_70b,      
    "GPT_OSS_120B": GROQ_GPT_120b, 
    "deterministic_python": PYTHON_DETERMINISTIC  
}


# allowed tasks
TASK_NUMERICAL_ANALYSIS  = "numerical_analysis"
TASK_BIO_ANALYSIS       = "bio_analysis"
TASK_SKILLS_ANALYSIS     = "skills_analysis"
TASK_VISUAL_BRAND_ANALYSIS = "visual_brand_analysis"
TASK_SUPER_AGENT         = "super_agent"

ALLOWED_EVALUATION_TASKS = [
    TASK_NUMERICAL_ANALYSIS,
    TASK_BIO_ANALYSIS,
    TASK_SKILLS_ANALYSIS,
    TASK_VISUAL_BRAND_ANALYSIS,
    TASK_SUPER_AGENT
]

# ------------------------------------------------------ Paths -------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_DATA_PATH = os.path.join(BASE_DIR, "assets", "eval_data", "eval_data.json")
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, "assets", "evaluation_results")
