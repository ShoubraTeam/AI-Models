from pydantic import BaseModel, Field
from typing import Literal


class SuperAgentResponse(BaseModel):
    verdict: Literal["accepted", "at_risk", "rejected"] = Field(
        description="Overall final verdict for the proposal."
    )

    summary_report: str = Field(
        description="A final clear summary about the overall proposal quality. Length should be 50-200 characters.",
        min_length=50,
        max_length=200
    )

    strengths_points: list[str] = Field(
        description="List of concrete strength points found in completed evaluator evidence.",
        default_factory=list
    )

    weakness_points: list[str] = Field(
        description="List of concrete weakness or rejection-risk points found in completed evaluator evidence.",
        default_factory=list
    )

    recommendations: list[str] = Field(
        description="List of direct recommendations that help the freelancer address proposal weaknesses.",
        default_factory=list
    )

    evaluation_limitations: list[str] = Field(
        description="List of unavailable evaluators or missing evidence that limits the final report.",
        default_factory=list
    )
