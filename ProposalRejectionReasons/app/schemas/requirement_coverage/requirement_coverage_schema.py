from pydantic import BaseModel, Field
from typing import List

# -------------------- Modifications
# --> adding summary
# -----------------------------------------
from ..schema_config import Summary
class RequirementCoverageSchema(BaseModel):
    details: str = Field(
        description="A clear, scientific justification explaining which requirement IDs were found, which were missing, and why."
    )
    requirements_covered_ids: List[str] = Field(
        description="List of exact original IDs (e.g., 'sh_req_1', 'bl_req_1') from the input list that are covered or satisfied by the freelancer proposal."
    )
    missing_requirements_ids: List[str] = Field(
        description="List of exact original IDs (e.g., 'sh_req_2', 'bl_req_2') from the input list that are missed or violated by the freelancer proposal."
    )

    summary: Summary