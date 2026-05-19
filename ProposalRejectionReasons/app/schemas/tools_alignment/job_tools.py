# ---------------------------------------------------------------   
# Schemas Controlling the output of the Job Tools Extractor Agent
# ---------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import Literal

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