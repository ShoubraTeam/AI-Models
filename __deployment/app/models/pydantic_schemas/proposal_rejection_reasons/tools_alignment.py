# ---------------------------------------------------------------   
# Schemas Controlling the output of the Job Tools Extractor Agent
# ---------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import Literal
from .schema_config import Summary

class JobTool(BaseModel):
    """A tool extracted from the job description"""
    tool_name: str = Field(
        description = "the name of the tool."
    )

    necessity_level: Literal["mandatory", "recommended", "optional", "forbidden"] = Field(
        description = "the necessity level of the tool."
    )

class JobToolResponse(BaseModel):
    """The Job Tool Extractor Response"""
    tools: list[JobTool] = Field(
        description = "The list of the tools mentioned in the job description & their necessity levels."
    )


class ProposalToolReview(BaseModel):
    """A tool extracted from the job description"""
    tool_name: str = Field(
        description = "the name of the tool."
    )

    necessity_level: Literal["mandatory", "recommended", "optional", "forbidden"] = Field(
        description = "the necessity level of the tool."
    )

    found_in_proposal: bool = Field(
        description = "a bool indicating if job_tool was mentioned in the propoal or not."
    )

    with_confidence: bool | None = Field(
        description = "a bool indicating if the freelancer mentioned it with confidence or not.",
        default = None
    )

class ProposalToolsResponse(BaseModel):
    """The Job Tool Extractor Response"""
    tool_reviews: list[ProposalToolReview] = Field(
        description = "The list of the reviews for each tool mentioned in job_tools_list."
    )

    summary: Summary