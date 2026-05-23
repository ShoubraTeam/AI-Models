from agents.BaseAgent import BaseAgent
from schemas.job_understanding.job_key_points_schema import JobKeyPointsSchema
from prompts.job_understanding.job_key_points_extraction_prompt import JOB_KEY_POINTS_EXTRACTION_PROMPT
from helpers.config import DEFAULT_MODELS_CFG


class JobKeyPointsExtractor(BaseAgent):
    """
    Sub-agent 1: Extracts core_problem, required_deliverables, and key_keywords
    from the job description.

    Designed to be tested and evaluated independently.

    Output: JobKeyPointsSchema
        - core_problem          : str
        - required_deliverables : List[str]
        - key_keywords          : List[str]  ← used by processing layer for keyword metrics
    """
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
            kwargs = DEFAULT_MODELS_CFG["job_key_points_extractor"]

        super().__init__(model_name, system_prompt, model_provider, tools, structured_response, **kwargs)

    
    def get_agent(self):
        return super().get_agent()
    
    def invoke(self, input, return_structured_op_only = True):
        return super().invoke(input, return_structured_op_only)
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)
    
