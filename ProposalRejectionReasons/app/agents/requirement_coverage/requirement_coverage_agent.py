from agents.BaseAgent import BaseAgent
from agents.requirement_coverage.job_requirements_extractor import JobRequirementsExtractor
from agents.requirement_coverage.job_requirements_matcher import JobRequirementsMatcher
from prompts.groq_native_prompts import REQUIREMENT_EXTRACTOR_PROMPT, REQUIREMENT_MATCHER_PROMPT
from schemas import ExtractedRequirementsSchema
from schemas.requirement_coverage.requirement_coverage_schema import RequirementCoverageSchema
import helpers.config as CFG

class RequirementCoverageAgent(BaseAgent):

    def __init__(self, extractor_model: str = CFG.GROQ_LLAMA_70b, matcher_model: str = CFG.GROQ_LLAMA_70b):
        super().__init__(
            model_name=matcher_model,
            system_prompt="Orchestrator Pipeline",
        )
        self.extractor_model = extractor_model
        self.matcher_model = matcher_model
        
        self.extractor = JobRequirementsExtractor(
            model_name=self.extractor_model,
            system_prompt=REQUIREMENT_EXTRACTOR_PROMPT,
            structured_response=ExtractedRequirementsSchema,
        )
        self.matcher = JobRequirementsMatcher(
            model_name=self.matcher_model,
            system_prompt=REQUIREMENT_MATCHER_PROMPT,
            structured_response=RequirementCoverageSchema,
        )

    def invoke(self, job_description: str, proposal_text: str) -> RequirementCoverageSchema:
        
        extraction_result = self.extractor.invoke(input=job_description)
        extracted_reqs = extraction_result.requirements
        
        final_evaluation = self.matcher.invoke(
            job_requirements=extracted_reqs,
            proposal_text=proposal_text
        )
        
        return final_evaluation
