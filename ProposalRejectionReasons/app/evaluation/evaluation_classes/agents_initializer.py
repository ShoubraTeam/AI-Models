# ---------------------------------------------------------------------
# A utility class used to init the agents required for a specific task
# ---------------------------------------------------------------------

# agents
from agents import JobToolsExtractor, ProposalToolsAnalyzer
from agents import JobRequirementsExtractor, JobRequirementsMatcher
from agents import JobKeyPointsExtractor, JobUnderstandingEvaluator
from agents import ExperienceEvidenceAgent
from agents import LanguageClarityEvaluator


# response schemas
from schemas import JobToolResponse, ProposalToolsResponse
from schemas import ExtractedRequirementsSchema, RequirementCoverageSchema
from schemas import JobKeyPointsSchema, JobUnderstandingEvalSchema
from schemas import ExperienceEvidenceSchema
from schemas import LanguageClarityEvalSchema

# system prompts
from prompts import JOB_TOOLS_EXTRACTION_PROMPT, PROPOSAL_TOOLS_EXTRACTION_PROMPT
from prompts import REQUIREMENT_EXTRACTOR_PROMPT, REQUIREMENT_MATCHER_PROMPT
from prompts import JOB_KEY_POINTS_EXTRACTION_PROMPT, JOB_UNDERSTANDING_EVALUATOR_PROMPT
from prompts import EXPERIENCE_EVIDENCE_PROMPT
from prompts import LANGUAGE_CLARITY_EVALUATOR_PROMPT

class AgentsInitializer:
    """
    A class used to init the agents required to evaluate them on a specific task
    """

    def __init__(self):
        pass

    # tools alignment
    @staticmethod
    def get_job_tool_extractor_agent(model_name, **kwargs):
        return JobToolsExtractor(
            model_name          = model_name,
            system_prompt       = JOB_TOOLS_EXTRACTION_PROMPT,
            structured_response = JobToolResponse,
            **kwargs
        )

    @staticmethod
    def get_proposal_tool_analyzer_agent(model_name, **kwargs):
        return ProposalToolsAnalyzer(
            model_name          = model_name,
            system_prompt       = PROPOSAL_TOOLS_EXTRACTION_PROMPT,
            structured_response = ProposalToolsResponse,
            **kwargs
        )

    # requirements coverage
    @staticmethod
    def get_job_requirements_extractor_agent(model_name, **kwargs):
        return JobRequirementsExtractor(
            model_name          = model_name,
            system_prompt       = REQUIREMENT_EXTRACTOR_PROMPT,
            structured_response = ExtractedRequirementsSchema,
            **kwargs
        )

    @staticmethod
    def get_job_requirements_matcher_agent(model_name, **kwargs):
        return JobRequirementsMatcher(
            model_name          = model_name,
            system_prompt       = REQUIREMENT_MATCHER_PROMPT,
            structured_response = RequirementCoverageSchema,
            **kwargs
        )

    # job understanding
    @staticmethod
    def get_job_key_points_extractor_agent(model_name, **kwargs):
        return JobKeyPointsExtractor(
            model_name          = model_name,
            system_prompt       = JOB_KEY_POINTS_EXTRACTION_PROMPT,
            structured_response = JobKeyPointsSchema,
            **kwargs
        )

    @staticmethod
    def get_job_understanding_evaluator_agent(model_name, **kwargs):
        return JobUnderstandingEvaluator(
            model_name          = model_name,
            system_prompt       = JOB_UNDERSTANDING_EVALUATOR_PROMPT,
            structured_response = JobUnderstandingEvalSchema,
            **kwargs
        )
    
    # evidence of experience
    @staticmethod
    def get_experience_evidence_finder_agent(model_name, **kwargs):
        return ExperienceEvidenceAgent(
            model_name          = model_name,
            system_prompt       = EXPERIENCE_EVIDENCE_PROMPT,
            structured_response = ExperienceEvidenceSchema,
            **kwargs
        )
    
    # language clarity
    @staticmethod
    def get_language_clarity_evaluator_agent(model_name, **kwargs):
        return LanguageClarityEvaluator(
            model_name          = model_name,
            system_prompt       = LANGUAGE_CLARITY_EVALUATOR_PROMPT,
            structured_response = LanguageClarityEvalSchema,
            **kwargs
        )
    
    # super-agent
    @staticmethod
    def get_super_agent(model_name, **kwargs):
        pass


    