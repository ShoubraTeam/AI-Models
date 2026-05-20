# ----------------------------------------------------------------
# General Configurations
# ----------------------------------------------------------------


# Models
PROVIDER_GROQ= "groq"
GROQ_LLAMA_8b = "llama-3.1-8b-instant"
GROQ_LLAMA_70b = "llama-3.3-70b-versatile"
GROQ_QWEN_32b = "qwen/qwen3-32b"
GROQ_GPT_120b = "openai/gpt-oss-120b"

PROVIDER_GOOGLE_GENAI = "google_genai"
GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
GEMINI_FLASH = "gemini-2.5-flash"


# Models CFG
MODELS_CFG = {
    "tools_alignment_pipeline" : {
        "job_tools_extractor_temperature" : 0.0,
        "job_tools_extractor_max_tokens"  : 512,
        "proposal_tools_analyzer_temperature"  : 0.0,
        "proposal_tools_analyzer_max_tokens"  : 1024
    }
}