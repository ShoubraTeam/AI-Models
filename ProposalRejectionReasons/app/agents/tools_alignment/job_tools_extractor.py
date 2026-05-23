# -----------------------------------------------------------------
# Implementing an Agent to extract tools from the job description
# -----------------------------------------------------------------


from ..BaseAgent import BaseAgent
# from schemas import JobTool, JobToolResponse
# from typing import get_args
from helpers.config import DEFAULT_MODELS_CFG

class JobToolsExtractor(BaseAgent):
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
            kwargs = DEFAULT_MODELS_CFG['job_tools_extractor']

        super().__init__(model_name, system_prompt, model_provider, tools, structured_response, **kwargs)
    
    def get_agent(self):
        return super().get_agent()
    
    def invoke(self, input, return_structured_op_only = True):
        return super().invoke(input, return_structured_op_only)
    # -------------------------------------------------------------------------------
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)

        
