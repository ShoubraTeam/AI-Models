from pydantic import BaseModel, Field
from typing import List, Literal

from ..schema_config import model_config

class RequirementItem(BaseModel):
    """An atomic client requirement extracted from the job description."""
    model_config = model_config
    id: str = Field(
        description=(
            "Unique sequential requirement identifier formatted as REQ_1, "
            "REQ_2, REQ_3, and so on."
        )
    )
    text: str = Field(
        description=(
            "Clean atomic functional requirement, feature, deliverable, or "
            "constraint requested by the client."
        )
    )
    necessity_level: Literal["mandatory", "recommended", "optional", "forbidden"] = Field(
        description=(
            "Client priority level: mandatory for required items, recommended "
            "for strong preferences, optional for extras, and forbidden for explicit exclusions."
        )
    )


class ExtractedRequirementsSchema(BaseModel):
    """List of requirements extracted from a job description."""
    model_config = model_config
    requirements: List[RequirementItem] = Field(
        description="Atomic functional requirements extracted from the job description."
    )
