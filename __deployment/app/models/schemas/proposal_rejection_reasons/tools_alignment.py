# ---------------------------------------------------------------   
# Schemas Controlling the output of the Job Tools Extractor Agent
# ---------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import Literal
from .schema_config import Summary, model_config


class JobTool(BaseModel):
    """A tool, platform, framework, or library mentioned in the job description."""
    model_config = model_config
    tool_name: str = Field(
        description="Canonical name of the tool as mentioned or clearly implied in the job description."
    )

    necessity_level: Literal["mandatory", "recommended", "optional", "forbidden"] = Field(
        description=(
            "Client's requirement level for the tool: mandatory, recommended, "
            "optional, or forbidden."
        )
    )


class JobToolResponse(BaseModel):
    """Tools extracted from a job description."""
    model_config = model_config
    tools: list[JobTool] = Field(
        description="All tools explicitly mentioned in the job description with their necessity levels."
    )




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