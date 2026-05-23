from agents.BaseAgent import BaseAgent
from schemas.job_understanding.job_key_points_schema import JobKeyPointsSchema
from prompts.job_understanding.job_key_points_extraction_prompt import JOB_KEY_POINTS_EXTRACTION_PROMPT
import helpers.config as CFG


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

    def __init__(self, model_name: str = CFG.GROQ_LLAMA_70b, temperature: float = None):

        if temperature is None:
            temperature = CFG.MODELS_CFG["job_understanding_pipeline"]["job_understanding_extractor_temperature"]

        max_tokens = CFG.MODELS_CFG["job_understanding_pipeline"]["job_understanding_extractor_max_tokens"]

        super().__init__(
            model_name=model_name,
            system_prompt=JOB_KEY_POINTS_EXTRACTION_PROMPT,
            model_provider=CFG.PROVIDER_GROQ,
            structured_response=JobKeyPointsSchema,
            temperature=temperature,
            max_tokens=max_tokens
        )
