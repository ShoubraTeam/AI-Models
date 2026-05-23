from agents.BaseAgent import BaseAgent
from schemas.job_understanding.job_understanding_eval_schema import JobUnderstandingEvalSchema
from prompts.job_understanding.job_understanding_evaluator_prompt import JOB_UNDERSTANDING_EVALUATOR_PROMPT
import helpers.config as CFG
from typing import List


class JobUnderstandingEvaluator(BaseAgent):
    """
    Sub-agent 2: Evaluates the proposal against the extracted job key points.

    Very task-specific — answers exactly 3 boolean questions:
        - problem_identified
        - solution_proposed
        - practical_steps_mentioned

    Everything else (keyword matching, scoring, similarity) is handled
    by the processing layer using normal code metrics.

    Designed to be tested and evaluated independently.

    Output: JobUnderstandingEvalSchema
        - problem_identified        : bool
        - solution_proposed         : bool
        - practical_steps_mentioned : bool
        - summary                   : str
        - confidence_score          : float
    """

    def __init__(self, model_name: str = CFG.GROQ_LLAMA_70b, temperature: float = None):

        if temperature is None:
            temperature = CFG.MODELS_CFG["job_understanding_pipeline"]["job_understanding_evaluator_temperature"]

        max_tokens = CFG.MODELS_CFG["job_understanding_pipeline"]["job_understanding_evaluator_max_tokens"]

        super().__init__(
            model_name=model_name,
            system_prompt=JOB_UNDERSTANDING_EVALUATOR_PROMPT,
            model_provider=CFG.PROVIDER_GROQ,
            structured_response=JobUnderstandingEvalSchema,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def invoke(
        self,
        core_problem: str,
        required_deliverables: List[str],
        proposal_text: str
    ) -> JobUnderstandingEvalSchema:
        """
        Note: key_keywords are NOT passed here intentionally.
        Keyword matching is done in the processing layer via set operations,
        not by the LLM — this keeps the agent focused and reduces token usage.

        Args:
            core_problem          : Extracted core problem from JobKeyPointsExtractor.
            required_deliverables : Extracted deliverables from JobKeyPointsExtractor.
            proposal_text         : The freelancer's proposal text.
        """
        formatted_input = (
            f"Core Problem:\n{core_problem}\n\n"
            f"Required Deliverables:\n{required_deliverables}\n\n"
            f"Freelancer Proposal:\n{proposal_text}"
        )
        return super().invoke(input=formatted_input)
