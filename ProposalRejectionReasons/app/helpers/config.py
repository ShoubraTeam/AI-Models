# ----------------------------------------------------------------
# General Configurations
# ----------------------------------------------------------------

import os

# Models
PROVIDER_GROQ         = "groq"
GROQ_LLAMA_8b         = "llama-3.1-8b-instant"
GROQ_LLAMA_70b        = "llama-3.3-70b-versatile"
GROQ_QWEN_32b         = "qwen/qwen3-32b"
GROQ_GPT_120b         = "openai/gpt-oss-120b"
GROQ_GPT_20b          = "openai/gpt-oss-20b"

PROVIDER_GOOGLE_GENAI = "google_genai"
GEMINI_FLASH_LITE     = "gemini-2.5-flash-lite"
GEMINI_FLASH          = "gemini-2.5-flash"



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

    "super_agent": {
        "temperature": 0.0,
        "max_tokens" : 1024
    }
}

# --------------------------------------------------- Scoring ----------------------------------------------------

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
TOOL_ALIGNMENT_ACCEPTANCE_THRESHOLD = 0.5
JOB_UNDERSTANDING_THRESHOLD = 0.5


# printing utils
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"




# ------------------------------------------------------ Evaluation CFG -------------------------------------------
# model_name --> provider model_name mapping
EVALUATION_MODELS_MAPPING = {
    "LLAMA_8B"         : GROQ_LLAMA_8b, 
    "GPT_OSS_20B"      : GROQ_GPT_20b,
    "QWEN_32B"         : GROQ_QWEN_32b,
    "LLAMA_70B"        : GROQ_LLAMA_70b,
    "GPT_OSS_120B"     : GROQ_GPT_120b,
}


# allowed tasks
TASK_JOB_TOOLS_EXTRACTOR         ="job_tools_extractor"
TASK_PROPOSAL_TOOLS_ANALYZER     = "proposal_tools_analyzer"

TASK_JOB_REQUIREMENTS_EXTRACTOR  = "job_requirements_extractor"
TASK_JOB_REQUIREMENTS_MATCHER    = "job_requirements_matcher"

TASK_JOB_KEY_POINTS_EXTRACTOR    = "job_key_points_extractor"
TASK_JOB_UNDERSTANDING_EVALUATOR = "job_understanding_evaluator"

TASK_EXPERIENCE_EVIDENCE_FINDER  = "experience_evidence_finder"
TASK_LANGUAGE_CLARITY_EVALUATOR  ="language_clarity_evaluator"

ALLOWED_EVALUATION_TASKS = [
    TASK_JOB_TOOLS_EXTRACTOR,
    TASK_PROPOSAL_TOOLS_ANALYZER,
    TASK_JOB_REQUIREMENTS_EXTRACTOR,
    TASK_JOB_REQUIREMENTS_MATCHER,
    TASK_JOB_KEY_POINTS_EXTRACTOR,
    TASK_JOB_UNDERSTANDING_EVALUATOR,
    TASK_EXPERIENCE_EVIDENCE_FINDER,
    TASK_LANGUAGE_CLARITY_EVALUATOR,
]

# ------------------------------------------------------ Paths -------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_DATA_PATH = os.path.join(BASE_DIR, "assets", "eval_data", "eval_data.json")
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, "assets", "evaluation_results")



