# ---------------------------------------------------------------
# Pydantic Schemas for logging saving results
# ---------------------------------------------------------------

from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Literal
from models.data_config import (
    JOB_DESC_TOOLS_DETECTION,
    JOB_DESC_TOOLS_RECOMMENDATION,
    JOB_DESC_JOB_DESCRIPTION_ENHANCEMENT
)

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

    task: Literal[
        "identity_recognition",
        "job_desc_tools_detection",
        "job_desc_tools_recommendation",
        "job_desc_job_description_enhancement",
        "PRR_job_features_extraction",
        "PRR_proposal_analysis",
        "profile_scorer_features_extraction",
        "profile_scorer_final_analysis"
        
        "RS_freelancer_embedding",
        "RS_job_embedding",
        # other feature tasks
    ]

    user_input: str | None | tuple | dict | Any = Field(
        None,
        description = "Input may be str (for LLM-based features) or Image for Identity Recognition & Profile analysis"
    )

    images: tuple[ImageLog, ImageLog] | None = None
    agent_output: str | bool | list[str] | dict[str, Any] | None = Field(
        description = "Output may be str, verified or not for Identity Recognition, or .. for Proposal Rejection Reasons"
    )

    duration_s: float = Field(description = "the inference time in seconds")
    user_feedback: Literal["Good", "Bad"] | str | None = None




