from agents.BaseAgent import BaseAgent
from schemas import SkillsAnalyzerSchema  
from helpers.config import DEFAULT_MODELS_CFG
from typing import List


class SkillsAnalyzer(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG.get("skills_analyzer", {"temperature": 0.1})

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)


    def invoke(
        self,
        declared_skills: List[str],
        job_role: str
    ) -> SkillsAnalyzerSchema:
        
        skills_string = ", ".join(declared_skills)
        
        formatted_input = (
            f"Target Job Role to Match: {job_role}\n"
            f"Freelancer Declared Skills List: [{skills_string}]"
        )
        return super().invoke(input=formatted_input)
    
    def get_agent(self):
        return super().get_agent()
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)