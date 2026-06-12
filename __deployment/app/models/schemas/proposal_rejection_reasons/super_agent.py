from pydantic import BaseModel, Field
from typing import Literal

from .schema_config import model_config


class PRR_SuperAgentResponse(BaseModel):
    """Final proposal assessment synthesized from completed sub-agent evidence."""

    model_config = model_config

    verdict: Literal["accepted", "at_risk", "rejected"] = Field(
        description="Final overall proposal verdict based on completed evaluator evidence."
    )

    summary_report: str = Field(
        description="Concise final judgment explaining the main acceptance or rejection risk.",
        min_length=1,
        max_length=500,
    )

    strengths_points: list[str] = Field(
        description="Concrete strengths supported by completed evaluator evidence. Return an empty list when none are supported."
    )

    weakness_points: list[str] = Field(
        description="Concrete weaknesses or rejection risks supported by completed evaluator evidence. Return an empty list when none are supported."
    )

    recommendations: list[str] = Field(
        description="Direct actions the freelancer can take to improve future proposals. Return an empty list when no recommendation is available."
    )

    evaluation_limitations: list[str] = Field(
        description="Unavailable evaluators, missing evidence, or caveats that limit the final judgment. Return an empty list when there are no limitations."
    )
