# ----------------------------------------------------------------
# General Configurations
# ----------------------------------------------------------------

import os

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
    "experience_evidence_agent" : {
        "temperature": 0.0,
        "max_tokens" : 1024
    },
    
    "language_clarity_evaluator": {
        "temperature": 0.0,
        "max_tokens" : 512
    },
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
JOB_UNDERSTANDING_THRESHOLD = 5.0


# printing utils
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"




# ------------------------------------------------------ Evaluation CFG -------------------------------------------
# model_name --> provider model_name mapping
EVALUATION_MODELS_MAPPING = {
    "LLAMA_8B"         : GROQ_LLAMA_8b, 
    "GEMINI_FLASH_LITE": GEMINI_FLASH_LITE,
    "GPT_OSS_20B"      : GROQ_GPT_20b,
    "QWEN_32B"         : GROQ_QWEN_32b,
    "GEMINI_FLASH"     : GEMINI_FLASH,
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

# paths
EVAL_DATA_PATH = "/mnt/d/Education/College/______GraduationProject/AI-Models/ProposalRejectionReasons/app/assets/eval_data/eval_data.json"
EVAL_RESULTS_PATH = "/mnt/d/Education/College/______GraduationProject/AI-Models/ProposalRejectionReasons/app/assets/evaluation_results"
