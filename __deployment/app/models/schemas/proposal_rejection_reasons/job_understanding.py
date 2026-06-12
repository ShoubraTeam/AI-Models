from pydantic import BaseModel, Field
from typing import List
from .schema_config import Summary, model_config


class JobKeyPointsSchema(BaseModel):
    """Core job understanding extracted from a freelance job description."""
    model_config = model_config

    core_problem: str = Field(
        description=(
            "Primary business problem, goal, or outcome the client wants the "
            "freelancer to address."
        )
    )

    required_deliverables: List[str] = Field(
        description=(
            "Concrete deliverables, outputs, or completed work products the "
            "client expects from the freelancer."
        )
    )

    key_keywords: List[str] = Field(
        description=(
            "Domain-specific keywords from the job description "
            "excluding tools and technologies — focus on skills, "
            "methodologies, and domain terms (e.g. 'REST API design', "
            "'agile', 'data modeling'). Tools are handled separately."
        )
    )



class JobUnderstandingEvalSchema(BaseModel):
    """
    Proposal evaluation focused on whether the freelancer understood the job.

    The fields capture LLM judgments about problem recognition, solution quality,
    actionable planning, and semantic keyword coverage. Numerical scoring is
    handled later by the processing layer.
    """
    model_config = model_config

    problem_identified: bool = Field(
        description=(
            "Whether the proposal clearly recognizes the client's core problem "
            "or desired outcome."
        )
    )
    solution_proposed: bool = Field(
        description=(
            "Whether the proposal offers a concrete, relevant solution rather "
            "than only generic availability or skill claims."
        )
    )
    practical_steps_mentioned: bool = Field(
        description=(
            "Whether the proposal describes actionable steps, implementation "
            "approach, workflow, or next actions."
        )
    )
    matched_keywords: List[str] = Field(
        description=(
            "Original job keywords or concepts that are explicitly mentioned "
            "or semantically implied in the proposal. Return an empty list when none match."
        )
    )
    missing_keywords: List[str] = Field(
        description=(
            "Original job keywords or concepts that are not mentioned, covered, "
            "or semantically implied in the proposal. Return an empty list when none are missing."
        )
    )
    summary: Summary
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the evaluation from 0.0 to 1.0.",
    )

