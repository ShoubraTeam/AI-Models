# ----------------------------------------------------------------
# General Configurations
# ----------------------------------------------------------------
# Models
PROVIDER_GROQ         = "groq"
GROQ_LLAMA_8b         = "llama-3.1-8b-instant"
GROQ_LLAMA_70b        = "llama-3.3-70b-versatile"
GROQ_QWEN_32b         = "qwen/qwen3-32b"
GROQ_GPT_120b         = "openai/gpt-oss-120b"
PROVIDER_GOOGLE_GENAI = "google_genai"
GEMINI_FLASH_LITE     = "gemini-2.5-flash-lite"
GEMINI_FLASH          = "gemini-2.5-flash"

# Models CFG
DEFAULT_MODELS_CFG = {
    "job_tools_extractor" : {
        "temperature" : 0.0,
        "max_tokens"  : 512
    },

    "proposal_tools_analyzer" : {
        "temperature" : 0.0,
        "max_tokens"  : 512
    },

    "job_key_points_extractor" : {
        "temperature": 0.0,
        "max_tokens" : 512,
    },

    "job_understanding_evaluator" : {
        "temperature": 0.0,
        "max_tokens" : 1024
    },


    "job_requirements_extractor" : {
        "temperature": 0.0,
        "max_tokens" : 512
    },

    "job_requirements_matcher" : {
        "temperature": 0.0,
        "max_tokens" : 1024
    },
}

# Tools Alignment Scoring
NECESSITY_LEVEL_WEIGHTS = {
    "mandatory"  : 1,
    "forbidden"  : -1,
    "recommended": 0.7,
    "optional"   : 0.5
}


WITH_CONFIDENCE_TOOL_WEIGHT = 1
GENERIC_TOOL_WEIGHT         = 0.5

# Job Understanding Scoring
JOB_UNDERSTANDING_THRESHOLD = 5.0


# printing utils
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"