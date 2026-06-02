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
        "temperature" : 0.2, # محتاجين كرييتفتي وسياق بشري أعلى هنا في الكتابة والملخص
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
    "LLAMA_8B"         : GROQ_LLAMA_8b, 
    "GEMINI_FLASH_LITE": GROQ_LLAMA_70b,
    "GPT_OSS_20B"      : GROQ_GPT_20b,
    "QWEN_32B"         : GROQ_QWEN_32b,
    "GEMINI_FLASH"     : GROQ_GPT_120b,
    "LLAMA_70B"        : GEMINI_FLASH_LITE,
    "GPT_OSS_120B"     : GEMINI_FLASH,
}


# TOOLS_ALIGNMENT_TASK        = "tools_alignment"
# REQUIREMENT_COVERAGE_TASK   = "requirement_coverage"
# JOB_UNDERSTANDING_TASK      = "job_understanding"
# LANGUAGE_CLARITY_TASK       = "language_clarity"
# EVIDENCE_OF_EXPERIENCE_TASK = "evidence_of_experience"
# SUPER_AGENT_TASK            = "super_agent"

# ALLOWED_EVALUATION_TASKS = [
#     TOOLS_ALIGNMENT_TASK,
#     REQUIREMENT_COVERAGE_TASK,
#     JOB_UNDERSTANDING_TASK,
#     LANGUAGE_CLARITY_TASK,
#     EVIDENCE_OF_EXPERIENCE_TASK,
#     SUPER_AGENT_TASK
# ]

# EVAL_DATA_PATH = "/mnt/d/Education/College/______GraduationProject/AI-Models/ProposalRejectionReasons/eval_data"