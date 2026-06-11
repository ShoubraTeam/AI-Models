from pydantic import BaseModel, Field
from typing import List

from ..schema_config import model_config
class JobKeyPointsSchema(BaseModel):
    """Core job understanding extracted from a freelance job description."""
    model_config = model_config

    core_problem: str = Field(
        description=(
            "Primary business problem, goal, or outcome the client wants the "
            "freelancer to address."
        )
    )

    required_deliverables: List[str] = Field(
        description=(
            "Concrete deliverables, outputs, or completed work products the "
            "client expects from the freelancer."
        )
    )
