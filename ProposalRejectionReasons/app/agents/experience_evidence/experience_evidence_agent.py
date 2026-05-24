from agents.BaseAgent import BaseAgent
from schemas.experience_evidence import ExperienceEvidenceSchema
from helpers.config import DEFAULT_MODELS_CFG

class ExperienceEvidenceAgent(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["experience_evidence_agent"]

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)

    def get_agent(self):
        return super().get_agent()
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)

    def invoke(self, job_desc: str, proposal_text: str) -> ExperienceEvidenceSchema:
        formatted_input = f"Job Description:\n{job_desc}\n\nFreelancer Proposal Text:\n{proposal_text}"
        return super().invoke(input=formatted_input)