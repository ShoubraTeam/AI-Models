# ---------------------------------------------------------------
# Schemas controlling the output of the Job Tools Extractor agent
# ---------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import Literal

from ..schema_config import model_config
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
