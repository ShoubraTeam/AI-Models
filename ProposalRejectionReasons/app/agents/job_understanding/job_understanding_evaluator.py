from agents.BaseAgent import BaseAgent
from schemas.job_understanding.job_understanding_schema import JobUnderstandingSchema
from prompts.job_understanding.job_understanding_evaluator_prompt import JOB_UNDERSTANDING_EVALUATOR_PROMPT
import helpers.config as CFG
from typing import List


class JobUnderstandingEvaluator(BaseAgent):
    """
    Sub-agent 2: Evaluates how well the freelancer understood the job,
    given the extracted key points and the proposal text.
    """

    def __init__(self, model_name: str = CFG.GROQ_LLAMA_70b, temperature: float = None):

        if temperature is None:
            temperature = CFG.MODELS_CFG["job_understanding_pipeline"]["job_understanding_evaluator_temperature"]

        max_tokens = CFG.MODELS_CFG["job_understanding_pipeline"]["job_understanding_evaluator_max_tokens"]

        super().__init__(
            model_name=model_name,
            system_prompt=JOB_UNDERSTANDING_EVALUATOR_PROMPT,
            model_provider=CFG.PROVIDER_GROQ,
            structured_response=JobUnderstandingSchema,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def invoke(
        self,
        core_problem: str,
        required_deliverables: List[str],
        key_keywords: List[str],
        proposal_text: str
    ) -> JobUnderstandingSchema:

        formatted_input = (
            f"Core Problem:\n{core_problem}\n\n"
            f"Required Deliverables:\n{required_deliverables}\n\n"
            f"Key Keywords:\n{key_keywords}\n\n"
            f"Freelancer Proposal:\n{proposal_text}"
        )
        return super().invoke(input=formatted_input)
