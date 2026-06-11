from pydantic import BaseModel, Field
from typing import List

from ..schema_config import Summary, model_config


class ExtractedProject(BaseModel):
    """A concrete past project or portfolio item mentioned in the proposal."""

    model_config = model_config

    project_overview: str = Field(
        description=(
            "Concise one-sentence overview of the past project, including its "
            "main functionality and relevant technical context."
        )
    )
    relevance_analysis: str = Field(
        description=(
            "Brief explanation of how the past project relates to the current "
            "job description's domain, features, or delivery expectations."
        )
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Similarity score from 0.0 to 1.0, where 1.0 means the past "
            "project closely matches the current job and 0.0 means it is not relevant."
        ),
    )


class ExperienceEvidenceSchema(BaseModel):
    """Evidence that the proposal mentions specific relevant past work."""

    model_config = model_config

    has_experience_evidence: bool = Field(
        description=(
            "True when the proposal explicitly mentions at least one specific "
            "past project, case study, portfolio item, or hands-on system; "
            "false for generic experience claims only."
        )
    )

    extracted_projects: List[ExtractedProject] = Field(
        description=(
            "Concrete past projects extracted from the proposal. Return an empty "
            "list when has_experience_evidence is false."
        )
    )

    summary: Summary
