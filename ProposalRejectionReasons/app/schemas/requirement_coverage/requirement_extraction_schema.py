from pydantic import BaseModel, Field
from typing import List, Literal, Annotated

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
            "A single, highly concise functional requirement or constraint. "
            "CRITICAL: Maximum 10 words. Absolutely NO programming languages, frameworks, "
            "tools, or developer roles allowed. Keep features as single cohesive blocks."
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
    
    requirements: Annotated[
        List[RequirementItem], 
        Field(
            min_length=1,
            max_length=5, 
            description=(
                "A clean list of highly concise functional requirements. "
                "CRITICAL: Do NOT force 5 items if the text has fewer. Extract EXACTLY the number "
                "of real explicit features found. Quality and conciseness are highly enforced."
            )
        )
    ]