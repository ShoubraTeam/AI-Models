from agents.BaseAgent import BaseAgent
from schemas.requirement_coverage.requirement_coverage_schema import RequirementCoverageSchema
from prompts.requirement_coverage.job_requirements_matching_prompt import REQUIREMENT_MATCHER_PROMPT
import helpers.config as CFG
from typing import List

class JobRequirementsMatcher(BaseAgent):

    def __init__(self, model_name: str = CFG.GROQ_LLAMA_70b, temperature: float = None):
        
        if temperature is None:
            temperature = CFG.MODELS_CFG["requirement_coverage_pipeline"]["job_requirements_matcher_temperature"]
            
        max_tokens = CFG.MODELS_CFG["requirement_coverage_pipeline"]["job_requirements_matcher_max_tokens"]
        
        super().__init__(
            model_name=model_name,
            system_prompt=REQUIREMENT_MATCHER_PROMPT,
            model_provider=CFG.PROVIDER_GROQ,
            structured_response=RequirementCoverageSchema,
            temperature=temperature,
            max_tokens=max_tokens # تمرير الـ tokens للـ BaseAgent
        )

    def invoke(self, job_requirements: List[str], proposal_text: str) -> RequirementCoverageSchema:
        """تجهيز النصين في متغير واحد واستدعاء الـ invoke بتاعة الـ BaseAgent"""
        formatted_input = f"Job Requirements List:\n{job_requirements}\n\nFreelancer Proposal Text:\n{proposal_text}"
        return super().invoke(input=formatted_input)