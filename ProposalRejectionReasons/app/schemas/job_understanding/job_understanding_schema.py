from pydantic import BaseModel, Field
from typing import List, Optional


class JobUnderstandingSchema(BaseModel):
    """Schema for the job understanding evaluation output"""

    score: float = Field(
        description="Score from 0 to 10 representing how well the freelancer understood the job."
    )
    problem_identified: bool = Field(
        description="Whether the freelancer clearly identified the core problem stated in the job."
    )
    solution_proposed: bool = Field(
        description="Whether the freelancer proposed a concrete and relevant solution."
    )
    practical_steps_mentioned: bool = Field(
        description="Whether the freelancer mentioned practical/actionable steps to solve the problem."
    )
    matched_keywords: List[str] = Field(
        description="Key keywords from the job description that appeared in the proposal."
    )
    missing_keywords: List[str] = Field(
        description="Important keywords from the job description that were missing in the proposal."
    )
    irrelevant_content: Optional[str] = Field(
        default=None,
        description="Any content in the proposal that is off-topic or irrelevant to the job."
    )
    summary: str = Field(
        description="A short summary explaining the score and how well the freelancer understood the job."
    )
    confidence_score: float = Field(
        description="How confident the agent is in its evaluation. Between 0.0 and 1.0."
    )
