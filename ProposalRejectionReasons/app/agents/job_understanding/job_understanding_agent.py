from agents.BaseAgent import BaseAgent
from agents.job_understanding.job_key_points_extractor import JobKeyPointsExtractor
from agents.job_understanding.job_understanding_evaluator import JobUnderstandingEvaluator
from schemas.job_understanding.job_understanding_schema import JobUnderstandingSchema
import helpers.config as CFG


class JobUnderstandingAgent(BaseAgent):
    """
    Orchestrator agent for the Job Understanding pipeline.

    Pipeline:
        1. JobKeyPointsExtractor   -> extracts core_problem, deliverables, keywords from job description
        2. JobUnderstandingEvaluator -> scores the proposal against those extracted key points

    Usage:
        agent = JobUnderstandingAgent()
        result = agent.invoke(job_description=..., proposal_text=...)
    """

    def __init__(
        self,
        extractor_model: str = CFG.GROQ_LLAMA_70b,
        evaluator_model: str = CFG.GROQ_LLAMA_70b
    ):
        super().__init__(
            model_name=evaluator_model,
            system_prompt="Job Understanding Orchestrator Pipeline",
            model_provider=CFG.PROVIDER_GROQ
        )
        self.extractor = JobKeyPointsExtractor(model_name=extractor_model)
        self.evaluator = JobUnderstandingEvaluator(model_name=evaluator_model)

    def invoke(self, job_description: str, proposal_text: str) -> JobUnderstandingSchema:
        """
        Run the full job understanding pipeline.

        Args:
            job_description : The full job posting text.
            proposal_text   : The freelancer's proposal text.

        Returns:
            JobUnderstandingSchema with score, details, confidence, and summary.
        """
        # Step 1 — Extract key points from the job description
        extraction_result = self.extractor.invoke(input=job_description)

        # Step 2 — Evaluate proposal against the extracted key points
        evaluation_result = self.evaluator.invoke(
            core_problem=extraction_result.core_problem,
            required_deliverables=extraction_result.required_deliverables,
            key_keywords=extraction_result.key_keywords,
            proposal_text=proposal_text
        )

        return evaluation_result
