# ---------------------------------------------------------------
# Pydantic Schemas for logging saving results
# ---------------------------------------------------------------

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

model_cfg = ConfigDict(
    str_to_lower = True,
    validate_assignment = True,
    populate_by_name = True,
    extra = "forbid",
    strict = False
)


class ImageLog(BaseModel):
    """Logging an img metadata"""
    model_config = model_cfg

    filename    : str
    content_type: str   | None = None
    size_mbytes : float | None = None
    saved_path  : str   | None = None

class AgentInferenceResult(BaseModel):
    """
    Saving input / output of an agent
    """
    model_config = model_cfg


    images: list[ImageLog] | None = None

    user_input: str | None = Field(
        None,
        description = "Input may be str (for LLM-based features) or Image for Identity Recognition & Profile analysis"
    )

    agent_output: str | bool = Field(
        description = "Output may be str, verified or not for Identity Recognition, or .. for Proposal Rejection Reasons"
    )

    duration_s: float = Field(
        description = "the inference time in seconds"
    )

    user_feedback: Literal["Good", "Bad"] | str | None = None





