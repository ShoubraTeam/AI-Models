# ---------------------------------------------------------------------------------------
# The final schema returned by subagents + post-processing after agent calls
# ---------------------------------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import Annotated
from ..schema_config import Summary, model_config

Score = Annotated[float, Field(ge=0.0, le=1.0)]
Reason = Annotated[str, Field(min_length=10, max_length=100)]


class FinalSubagentResult(BaseModel):
    """
    Normalized result produced by a sub-agent processing layer.

    Attributes:
        score: Task-level score normalized between 0.0 and 1.0.
        accepted: Whether the score passes the task-specific threshold.
        summary: Concise explanation of the task result.
        acceptance_reasons: Short reasons supporting acceptance when accepted is true; null otherwise.
        rejection_reasons: Short reasons explaining rejection risk when accepted is false; null otherwise.
    """
    model_config = model_config

    score: Score
    accepted: bool
    summary: Summary

    acceptance_reasons: list[Reason] | None = Field(
        description="Acceptance reasons when the sub-agent accepts the proposal; null when it rejects."
    )
    rejection_reasons: list[Reason] | None = Field(
        description="Rejection reasons when the sub-agent rejects the proposal; null when it accepts."
    )
