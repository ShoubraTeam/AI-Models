from pydantic import BaseModel, Field
from typing import List

class SkillsAnalyzerSchema(BaseModel):
    score: float = Field(
        description="Alignment score from 0.0 to 1.0 representing how well the declared skills cover the target job role."
    )
    missing_essential_skills: List[str] = Field(
        ..., 
        description="Strictly max 5 missing core tech/skill names only (e.g., ['Docker', 'Kubernetes'])."
    )
    irrelevant_skills: List[str] = Field(
        ..., 
        description="Strictly max 5 unrelated or diluting skill names only (e.g., ['Data Entry'])."
    )
