from pydantic import BaseModel, Field
from typing import List

class RequirementCoverageSchema(BaseModel):
    score: float = Field(
        description="The calculated coverage score strictly between 0.0 and 1.0, representing (len(requirements) / total_requirements)."
    )
    details: str = Field(
        description="A clear, scientific justification explaining which requirements were found, which were missing, and why."
    )
    requirements: List[str] = Field(
        description="List of the exact requirements from the job description that were successfully covered in the proposal."
    )
    missing_requirements: List[str] = Field(
        description="List of the exact requirements from the job description that were completely missed or ignored in the proposal."
    )