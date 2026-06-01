from pydantic import BaseModel, Field
from typing import List
from ..schema_config import Summary


class JobUnderstandingEvalSchema(BaseModel):
    """
    Output of JobUnderstandingEvaluator.

    ONLY contains decisions that require LLM reasoning.
    Keyword matching is done semantically by the LLM here —
    it understands that 'ML' == 'machine learning', 'Postgres' == 'PostgreSQL', etc.
    Coverage score and final scoring are calculated in processing.
    """
    problem_identified: bool = Field(
        description="Whether the freelancer clearly identified the core problem stated in the job."
    )
    solution_proposed: bool = Field(
        description="Whether the freelancer proposed a concrete and relevant solution."
    )
    practical_steps_mentioned: bool = Field(
        description="Whether the freelancer mentioned practical or actionable steps."
    )
    matched_keywords: List[str] = Field(
        description="Keywords from the job description that were mentioned or implied in the proposal. "
                    "Include semantic equivalents: 'ML' matches 'machine learning', "
                    "'Postgres' matches 'PostgreSQL', 'JS' matches 'JavaScript'."
    )
    missing_keywords: List[str] = Field(
        description="Keywords from the job description that had NO mention or equivalent "
                    "in the proposal. Be strict — only list truly absent concepts."
    )
    summary: Summary
    confidence_score: float = Field(
        description="How confident the agent is in its evaluation. Between 0.0 and 1.0."
    )
