from agents.BaseAgent import BaseAgent
from schemas import RequirementCoverageSchema
from helpers.config import DEFAULT_MODELS_CFG
from typing import List

class JobRequirementsMatcher(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        model_provider: str = "groq",
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["job_requirements_matcher"]

        super().__init__(model_name, system_prompt, model_provider, tools, structured_response, **kwargs)

    def get_agent(self):
        return super().get_agent()
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)

    def invoke(self, job_requirements: List[str], proposal_text: str) -> RequirementCoverageSchema:
        formatted_input = f"Job Requirements List:\n{job_requirements}\n\nFreelancer Proposal Text:\n{proposal_text}"
        return super().invoke(input=formatted_input)