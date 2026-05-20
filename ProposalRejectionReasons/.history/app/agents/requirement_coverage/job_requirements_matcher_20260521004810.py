from agents.BaseAgent import BaseAgent
from schemas.requirement_coverage.requirement_coverage_schema import RequirementCoverageSchema
from prompts.requirement_coverage.job_requirements_matching_prompt import REQUIREMENT_MATCHER_PROMPT
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import List

class JobRequirementsMatcher(BaseAgent):
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
        super().__init__(
            model_name=model_name,
            system_prompt=REQUIREMENT_MATCHER_PROMPT,
            model_provider="groq",
            structured_response=RequirementCoverageSchema,
            temperature=temperature
        )

    def invoke(self, job_requirements: List[str], proposal_text: str) -> RequirementCoverageSchema:
        formatted_input = f"Job Requirements List:\n{job_requirements}\n\nFreelancer Proposal Text:\n{proposal_text}"
        return super().invoke(input=formatted_input)