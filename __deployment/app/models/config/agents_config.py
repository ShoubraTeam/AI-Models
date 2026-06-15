# ---------------------------------------------- #
# Configurationss For all agents in the system
# ---------------------------------------------- #

from prompts import *
from models.schemas import (
    # PRR
    JobToolResponse,
    ProposalToolsResponse,

    JobKeyPointsSchema,
    JobUnderstandingEvalSchema,

    ExtractedRequirementsSchema,
    RequirementCoverageSchema,

    LanguageClarityEvalSchema,
    ExperienceEvidenceSchema,

    PRR_SuperAgentResponse,

    # PA
    VisualBrandEvaluationSchema,
    BioAnalyzerSchema,
    NumericalAnalyzerSchema,
    SkillsAnalyzerSchema,
    PA_SuperAgentSchema

)
# ----------------- ID Reco --------------------
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

CARD_CLASSIFICATION_MODEL = "google/siglip-base-patch16-224"

# ----------------- JD ENH --------------------
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


# ----------- PRR -----------------
PRR_DEFAULT_MODELS_CFG = {
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
        "max_tokens" : 512
    },

    "job_requirements_matcher" : {
        "temperature": 0.0,
        "max_tokens" : 512
    },
    "experience_evidence_agent" : {
        "temperature": 0.0,
        "max_tokens" : 512
    },
    
    "language_clarity_evaluator": {
        "temperature": 0.0,
        "max_tokens" : 1024
    },
}


PRR_NECESSITY_LEVEL_WEIGHTS = {
    "mandatory"  : 1,
    "forbidden"  : -1,
    "recommended": 0.7,
    "optional"   : 0.5
}


PRR_WITH_CONFIDENCE_TOOL_WEIGHT = 1
PRR_GENERIC_TOOL_WEIGHT         = 0.5


TA_TOOL_ALIGNMENT_THRESHOLD       = 0.5
JD_JOB_UNDERSTANDING_THRESHOLD    = 0.5
RQ_REQUIREMENT_COVERAGE_THRESHOLD = 0.5
LANGUAGE_CLARITY_THRESHOLD        = 0.5 
EXPERIENCE_EVIDENCE_THRESHOLD     = 0.5


TA_JOB_TOOLS_EXTRACTOR_CFG = {
    "model_name"         : "llama-3.1-8b-instant",
    "system_prompt"      : JOB_TOOLS_EXTRACTION_PROMPT,
    "structured_response": JobToolResponse,
    "temperature"        : 0.0,
    "max_tokens"         : 1024,
    "top_p"              : 0.9
}

TA_PROPOSAL_TOOLS_ANALYZER_CFG = {
    "model_name"         : "llama-3.1-8b-instant",
    "system_prompt"      : PROPOSAL_TOOLS_EXTRACTION_PROMPT,
    "structured_response": ProposalToolsResponse,
    "temperature"        : 0.0,
    "max_tokens"         : 1024,
    "top_p"              : 0.9
}

# Job Understanding (JD)
JD_JOB_KEY_POINTS_CFG = {
    "model_name"         : "llama-3.3-70b-versatile",
    "system_prompt"      : JOB_KEY_POINTS_EXTRACTION_PROMPT,
    "structured_response": JobKeyPointsSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 1024,
    "top_p"              : 0.9
}

JD_JOB_UNDERSTANDING_EVALUATOR_CFG = {
    "model_name"         : "llama-3.3-70b-versatile",
    "system_prompt"      : JOB_UNDERSTANDING_EVALUATOR_PROMPT,
    "structured_response": JobUnderstandingEvalSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 1024,
    "top_p"              : 0.9
}

# Requirement Coverage (RQ)
RQ_REQUIREMENT_EXTRACTOR_CFG = {
    "model_name"         : "llama-3.1-8b-instant",
    "system_prompt"      : REQUIREMENT_EXTRACTOR_PROMPT,
    "structured_response": ExtractedRequirementsSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

RQ_REQUIREMENT_COVERAGE_EVALUATOR_CFG = {
    "model_name"         : "llama-3.1-8b-instant",
    "system_prompt"      : REQUIREMENT_MATCHER_PROMPT,
    "structured_response": RequirementCoverageSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

# Language Clarity
LANGUAGE_CLARITY_EVALUATOR_CFG = {
    "model_name"         : "llama-3.3-70b-versatile",
    "system_prompt"      : LANGUAGE_CLARITY_EVALUATOR_PROMPT,
    "structured_response": LanguageClarityEvalSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 1024,
    "top_p"              : 0.9
}

# Evidence of experience
EVIDENCE_OF_EXPERIENCE_EVALUATOR_CFG = {
    "model_name"         : "llama-3.1-8b-instant",
    "system_prompt"      : EXPERIENCE_EVIDENCE_PROMPT,
    "structured_response": ExperienceEvidenceSchema,
    "temperature"        : 0.0,
    "max_tokens"         : 512,
    "top_p"              : 0.9
}

# Super Agent
PRR_SUPER_AGENT_CFG = {
    "model_name"         : "llama-3.3-70b-versatile",
    "system_prompt"      : SUPER_AGENT_SYSTEM_PROMPT,
    "structured_response": PRR_SuperAgentResponse,
    "temperature"        : 0.1,
    "max_tokens"         : 2048,
    "top_p"              : 0.9
}



# ----------- Profile Scorer Agents Configs -------------
PA_DEFAULT_MODELS_CFG = {
    "visual_brand_evaluator" : {
        "temperature" : 0.0,
        "max_tokens"  : 1024
    },
    "bio_analyzer" : {
        "temperature" : 0.0,
        "max_tokens"  : 1024
    },
    "skills_analyzer" : {
        "temperature" : 0.0,
        "max_tokens"  : 1024
    },
    "super_agent" : {
        "temperature" : 0.0, 
        "max_tokens"  : 1024
    }
}


# Visual Brand Analyzer
PROFILE_VISUAL_BRAND_CFG = {
    "model_name": "gemini-2.5-flash-lite",  
    "system_prompt": VISUAL_BRAND_PROMPT, 
    "structured_response": VisualBrandEvaluationSchema,  
    "temperature": 0.0,
    "max_tokens": 1024,
    "top_p": 0.9
}

# Bio Copywriting Analyzer
PROFILE_BIO_ANALYSIS_CFG = {
    "model_name": "llama-3.1-8b-instant",
    "system_prompt": BIO_ANALYZER_PROMPT,  
    "structured_response": BioAnalyzerSchema,  
    "temperature": 0.0,
    "max_tokens": 1024,
    "top_p": 0.9
}

# Technical Skills Analyzer
PROFILE_SKILLS_ANALYSIS_CFG = {
    "model_name": "gemini-2.5-flash-lite",
    "system_prompt": SKILLS_ANALYZER_PROMPT,  
    "structured_response": SkillsAnalyzerSchema,  
    "temperature": 0.0,
    "max_tokens": 1024,
    "top_p": 0.9
}

PROFILE_NUMERICAL_ANALYSIS_CFG = {
    "model_name": "deterministic_python",
    "system_prompt": None, 
    "structured_response": NumericalAnalyzerSchema,  
}

# Profile Scorer Master SuperAgent
PA_PROFILE_SUPER_AGENT_CFG = {
    "model_name": "openai/gpt-oss-120b",
    "system_prompt": SUPER_AGENT_PROMPT,  
    "structured_response": PA_SuperAgentSchema, 
    "temperature": 0.0,
    "max_tokens": 1024,
    "top_p": 0.9
}


# ----------------- RS --------------------
