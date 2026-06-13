# ----------------------------------------------------------------------
# Schemas controlling the output of the Proposal Tools Analyzer agent
# ----------------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import Literal

from ..schema_config import Summary, model_config


class ProposalToolReview(BaseModel):
    """Proposal coverage review for one tool from the job tools list."""

    model_config = model_config

    tool_name: str = Field(
        description="Original tool name from the provided job tools list."
    )

    necessity_level: Literal["mandatory", "recommended", "optional", "forbidden"] = Field(
        description="Original necessity level from the provided job tools list."
    )

    found_in_proposal: bool = Field(
        description="Whether the proposal mentions this tool or a clear semantic equivalent."
    )

    with_confidence: bool | None = Field(
        description=(
            "True when the proposal mentions the tool with confident, relevant context; "
            "false when mentioned only generically; null when the tool is not mentioned."
        )
    )


class ProposalToolsResponse(BaseModel):
    """Tool-alignment review for a freelancer proposal."""

    model_config = model_config

    tool_reviews: list[ProposalToolReview] = Field(
        description="Review result for every tool from the provided job tools list."
    )

    summary: Summary
