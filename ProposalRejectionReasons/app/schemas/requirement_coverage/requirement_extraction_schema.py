from pydantic import BaseModel, Field
from typing import List

class RequirementItem(BaseModel):
    id: str = Field(
        description="Unique identifier for the requirement, formatted strictly as REQ_1, REQ_2, REQ_3, etc."
    )
    text: str = Field(
        description="The clean, atomic core functional requirement or feature requested in the job description, excluding specific developer tools or languages."
    )

class ExtractedRequirementsSchema(BaseModel):
    requirements: List[RequirementItem] = Field(
        description="A structured list containing all the individual mandatory functional requirements, each paired with its unique sequential ID."
    )