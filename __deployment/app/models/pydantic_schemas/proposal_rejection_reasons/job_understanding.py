from pydantic import BaseModel, Field
from typing import List
from .schema_config import Summary

class JobKeyPointsSchema(BaseModel):
    """
    Output of JobKeyPointsExtractor.
    Extracts structured key points from the job description.
    Used downstream by the evaluator and by metric calculations.
    """
    core_problem: str = Field(
        description="The main problem or goal the client wants to solve."
    )
    required_deliverables: List[str] = Field(
        description="List of concrete deliverables or outcomes the client expects."
    )
    key_keywords: List[str] = Field(
        description="Domain-specific keywords from the job description "
                    "excluding tools and technologies — focus on skills, "
                    "methodologies, and domain terms (e.g. 'REST API design', "
                    "'agile', 'data modeling'). Tools are handled separately."
    )



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
        default_factory=list,
        description="Keywords from the job description that were mentioned or implied in the proposal. "
                    "Include semantic equivalents: 'ML' matches 'machine learning', "
                    "'Postgres' matches 'PostgreSQL', 'JS' matches 'JavaScript'."
    )
    missing_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords from the job description that had NO mention or equivalent "
                    "in the proposal. Be strict — only list truly absent concepts."
    )
    summary: Summary = "Evaluation completed."
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How confident the agent is in its evaluation. Between 0.0 and 1.0."
    )
