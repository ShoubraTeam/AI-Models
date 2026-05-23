# -----------------------------------------------------------------
# Implementing an Agent to extract tools from the job description
# -----------------------------------------------------------------


from ..BaseAgent import BaseAgent
# from schemas import JobTool, JobToolResponse
# from typing import get_args

from helpers.config import DEFAULT_MODELS_CFG

class ProposalToolsAnalyzer(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["proposal_tools_analyzer"]

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)
    
    def get_agent(self):
        return super().get_agent()
    
    def invoke(self, input, return_structured_op_only = True):
        return super().invoke(input, return_structured_op_only)

    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)

        
