from ..base_agent import BaseAgent
from agents.proposal_rejection_reasons.requirement_coverage.job_requirements_extractor import JobRequirementsExtractor
from agents.proposal_rejection_reasons.requirement_coverage.job_requirements_matcher import JobRequirementsMatcher
from models.pydantic_schemas import RequirementCoverageSchema
import helpers.config as CFG

class RequirementCoverageAgent(BaseAgent):

    def __init__(self, extractor_model: str = CFG.GROQ_LLAMA_70b, matcher_model: str = CFG.GROQ_LLAMA_70b):
        super().__init__(
            model_name=matcher_model,
            system_prompt="Orchestrator Pipeline",
            model_provider=CFG.PROVIDER_GROQ 
        )
        self.extractor_model = extractor_model
        self.matcher_model = matcher_model
        
        self.extractor = JobRequirementsExtractor(model_name=self.extractor_model)
        self.matcher = JobRequirementsMatcher(model_name=self.matcher_model)

    def invoke(self, job_description: str, proposal_text: str) -> RequirementCoverageSchema:
        
        extraction_result = self.extractor.invoke(input=job_description)
        extracted_reqs = extraction_result.job_requirements
        
        final_evaluation = self.matcher.invoke(
            job_requirements=extracted_reqs,
            proposal_text=proposal_text
        )
        
        return final_evaluation
