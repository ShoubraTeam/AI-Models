# ---------------------------------------------------------------
# Pydantic Schemas for logging saving results
# ---------------------------------------------------------------

from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Literal



model_cfg = ConfigDict(
    str_to_lower        = True,
    validate_assignment = True,
    populate_by_name    = True,
    extra               = "forbid",
    strict              = False
)


class ImageLog(BaseModel):
    """Saving img metadata"""
    model_config = model_cfg

    filename    : str
    content_type: str   | None = None
    size_mbytes : float | None = None
    saved_path  : str   | None = None




class AgentInput(BaseModel):
    """Saving Agent Input"""
    input_id : str = Field(description = "Identifier of the input [images - job_desc - ..]")

    value    : None | str | dict[str, Any] = Field(
        default = None,
        description = "The actual value"
    )

    images   : tuple[ImageLog, ImageLog] | None = Field(
        None,
        description = "Identity Recognition & Profile Photo Analysis input"
    )

class AgentOutput(BaseModel):
    """Saving Agent Output"""
    output_id : str  | list[str] = Field(description = "identifier of the output [image verification, ...]")

    value     : bool | str | list[str] | dict[str, Any] = Field(description = "The actual value")

class AgentResultsToSave(BaseModel):
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
        "profile_scorer_final_analysis",
        
        "RS_freelancer_embedding",
        "RS_job_embedding",
    ]

    agent_input: AgentInput
    agent_output: AgentOutput
    duration_s: float = Field(description = "the inference time in seconds")
    user_feedback: str | None = None



