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
MODELS_CFG = {
    "tools_alignment_pipeline": {
        "job_tools_extractor_temperature"      : 0.0,
        "job_tools_extractor_max_tokens"       : 512,
        "proposal_tools_analyzer_temperature"  : 0.0,
        "proposal_tools_analyzer_max_tokens"   : 1024
    },
    "requirement_coverage_pipeline": {
        "job_requirements_extractor_temperature": 0.0,
        "job_requirements_extractor_max_tokens" : 512,
        "job_requirements_matcher_temperature"  : 0.0,
        "job_requirements_matcher_max_tokens"   : 1024
    },
    "job_understanding_pipeline": {
        "job_understanding_extractor_temperature": 0.0,
        "job_understanding_extractor_max_tokens" : 512,
        "job_understanding_evaluator_temperature": 0.0,
        "job_understanding_evaluator_max_tokens" : 1024
    },
    "language_clarity_pipeline": {
    "language_clarity_evaluator_temperature": 0.0,
    "language_clarity_evaluator_max_tokens" : 512
    }
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
