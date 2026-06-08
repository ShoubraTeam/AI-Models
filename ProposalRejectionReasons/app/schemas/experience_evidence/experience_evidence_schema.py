from pydantic import BaseModel, Field
from typing import List

from ..schema_config import Summary
class ExtractedProject(BaseModel):
    project_overview: str = Field(
        description="A concise summary of the past project, including its core functionality and any key tools used within its context."
    )
    relevance_analysis: str = Field(
        description="A direct technical analysis explaining how this past project relates to the current Job Description."
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="A technical score between 0.0 and 1.0 evaluating how closely this past project matches the current Job Description context."
    )

class ExperienceEvidenceSchema(BaseModel):
    has_experience_evidence: bool = Field(
        description="True ONLY if the freelancer explicitly mentions specific past projects or hands-on built solutions. False if they only provide generic claims of years of experience without context."
    )
    extracted_projects: List[ExtractedProject] = Field(
        description="List of all validated past projects extracted from the proposal text. Must be empty if has_experience_evidence is False."
    )
    summary: Summary