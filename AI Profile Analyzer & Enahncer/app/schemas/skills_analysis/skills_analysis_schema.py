from pydantic import BaseModel, Field
from typing import List

class SkillsAnalyzerSchema(BaseModel):
    score: float = Field(
        description="Alignment score from 0.0 to 1.0 representing how well the declared skills cover the target job role."
    )
    missing_essential_skills: List[str] = Field(
        description="Critical core skills or technologies required for the job role that the freelancer completely missed."
    )
    irrelevant_skills: List[str] = Field(
        description="Skills listed by the freelancer that do not belong to the target role and cause profile dilution."
    )
