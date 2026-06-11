# ----------------------------
# The Pipeline CFG
# ----------------------------

from prompts.groq_native_prompts import JOB_TOOLS_EXTRACTION_PROMPT, PROPOSAL_TOOLS_EXTRACTION_PROMPT
from prompts.groq_native_prompts import JOB_KEY_POINTS_EXTRACTION_PROMPT, JOB_UNDERSTANDING_EVALUATOR_PROMPT
from prompts.groq_native_prompts import REQUIREMENT_EXTRACTOR_PROMPT, REQUIREMENT_MATCHER_PROMPT
from prompts.groq_native_prompts import EXPERIENCE_EVIDENCE_PROMPT
from prompts.groq_native_prompts import LANGUAGE_CLARITY_EVALUATOR_PROMPT
from prompts.groq_native_prompts import SUPER_AGENT_SYSTEM_PROMPT

from schemas import JobToolResponse, ProposalToolsResponse
from schemas import JobKeyPointsSchema, JobUnderstandingEvalSchema
from schemas import ExtractedRequirementsSchema, RequirementCoverageSchema
from schemas import ExperienceEvidenceSchema
from schemas import LanguageClarityEvalSchema
from schemas import SuperAgentResponse


# Tools Alignment (TA)
TA_TOOL_ALIGNMENT_THRESHOLD = 0.5
TA_JOB_TOOLS_EXTRACTOR_CFG = {
    "model_name": "llama-3.1-8b-instant",
    "system_prompt": JOB_TOOLS_EXTRACTION_PROMPT,
    "structured_response": JobToolResponse,
    "temperature": 0.0,
    "max_tokens": 1024,
    "top_p": 0.9,
}

TA_PROPOSAL_TOOLS_ANALYZER_CFG = {
    "model_name": "llama-3.1-8b-instant",
    "system_prompt": PROPOSAL_TOOLS_EXTRACTION_PROMPT,
    "structured_response": ProposalToolsResponse,
    "temperature": 0.0,
    "max_tokens": 1024,
    "top_p": 0.9,
}

# Job Understanding (JD)
JD_JOB_UNDERSTANDING_THRESHOLD = 0.5
JD_JOB_KEY_POINTS_CFG = {
    "model_name": "llama-3.3-70b-versatile",
    "system_prompt": JOB_KEY_POINTS_EXTRACTION_PROMPT,
    "structured_response": JobKeyPointsSchema,
    "temperature": 0.0,
    "max_tokens": 1024,
    "top_p": 0.9,
}

JD_JOB_UNDERSTANDING_EVALUATOR_CFG = {
    "model_name": "llama-3.3-70b-versatile",
    "system_prompt": JOB_UNDERSTANDING_EVALUATOR_PROMPT,
    "structured_response": JobUnderstandingEvalSchema,
    "temperature": 0.0,
    "max_tokens": 1024,
    "top_p": 0.9,
}

# Requirement Coverage (RQ)
RQ_REQUIREMENT_COVERAGE_THRESHOLD = 0.5
RQ_REQUIREMENT_EXTRACTOR_CFG = {
    "model_name": "llama-3.1-8b-instant",
    "system_prompt": REQUIREMENT_EXTRACTOR_PROMPT,
    "structured_response": ExtractedRequirementsSchema,
    "temperature": 0.0,
    "max_tokens": 2048,
    "top_p": 0.9,
}

RQ_REQUIREMENT_COVERAGE_EVALUATOR_CFG = {
    "model_name": "llama-3.3-70b-versatile",
    "system_prompt": REQUIREMENT_MATCHER_PROMPT,
    "structured_response": RequirementCoverageSchema,
    "temperature": 0.0,
    "max_tokens": 2048,
    "top_p": 0.9,
}

# Language Clarity
LANGUAGE_CLARITY_THRESHOLD = 0.5
LANGUAGE_CLARITY_EVALUATOR_CFG = {
    "model_name": "llama-3.3-70b-versatile",
    "system_prompt": LANGUAGE_CLARITY_EVALUATOR_PROMPT,
    "structured_response": LanguageClarityEvalSchema,
    "temperature": 0.0,
    "max_tokens": 1024,
    "top_p": 0.9,
}

# Evidence of experience
EXPERIENCE_EVIDENCE_THRESHOLD = 0.5
EVIDENCE_OF_EXPERIENCE_EVALUATOR_CFG = {
    "model_name": "llama-3.3-70b-versatile",
    "system_prompt": EXPERIENCE_EVIDENCE_PROMPT,
    "structured_response": ExperienceEvidenceSchema,
    "temperature": 0.0,
    "max_tokens": 1024,
    "top_p": 0.9,
}

# Super Agent
SUPER_AGENT_CFG = {
    "model_name": "llama-3.3-70b-versatile",
    "system_prompt": SUPER_AGENT_SYSTEM_PROMPT,
    "structured_response": SuperAgentResponse,
    "temperature": 0.1,
    "max_tokens": 2048,
    "top_p": 0.9,
}
