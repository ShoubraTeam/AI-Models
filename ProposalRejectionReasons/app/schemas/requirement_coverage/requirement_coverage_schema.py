from pydantic import BaseModel, Field
from typing import List

# -------------------- Modifications
# --> adding summary
# -----------------------------------------
from schemas import Summary
class RequirementCoverageSchema(BaseModel):
    details: str = Field(
        description="A clear, scientific justification explaining which requirement IDs were found, which were missing, and why."
    )
    requirements_covered_ids: List[str] = Field(
        description="List of the exact requirement IDs (e.g., 'REQ_1', 'REQ_2') that were successfully covered or addressed in the freelancer's proposal."
    )
    missing_requirements_ids: List[str] = Field(
        description="List of the exact requirement IDs (e.g., 'REQ_1', 'REQ_3') that were completely missed, ignored, or violated in the freelancer's proposal."
    )

    summary: Summary