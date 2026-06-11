from pydantic import BaseModel, Field
from typing import List, Literal
from .schema_config import Summary


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

    summary: Summary = "Requirements coverage evaluation completed."



class RequirementItem(BaseModel):
    id: str = Field(
        description="Unique identifier for the requirement, formatted strictly as REQ_1, REQ_2, REQ_3, etc. MAXIMUM 10 REQUIREMENTS."
    )
    text: str = Field(
        description="The clean, atomic core functional requirement or feature requested in the job description, excluding specific developer tools or languages."
    )
    necessity_level: Literal["mandatory", "recommended", "optional", "forbidden"] = Field(
        description="The priority or constraint level of the requirement based on client framing."
    )


class ExtractedRequirementsSchema(BaseModel):
    requirements: List[RequirementItem] = Field(
        description="A structured list containing all the individual mandatory functional requirements, each paired with its unique sequential ID."
    )
