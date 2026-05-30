from agents.BaseAgent import BaseAgent
from schemas import BioAnalyzerSchema 
from helpers.config import DEFAULT_MODELS_CFG


class BioAnalyzer(BaseAgent):

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["bio_analyzer"]

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)


    def invoke(
        self,
        bio_text: str,
        job_role: str
    ) -> BioAnalyzerSchema:

        formatted_input = (
            f"Freelancer Target Job Role: {job_role}\n"
            f"Freelancer Profile Bio/Summary Text:\n"
            f"\"\"\"\n{bio_text}\n\"\"\""
        )
        return super().invoke(input=formatted_input)
    
    def get_agent(self):
        return super().get_agent()
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)