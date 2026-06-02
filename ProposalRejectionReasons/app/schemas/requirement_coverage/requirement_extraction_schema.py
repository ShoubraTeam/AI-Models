from pydantic import BaseModel, Field
from typing import List, Literal


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