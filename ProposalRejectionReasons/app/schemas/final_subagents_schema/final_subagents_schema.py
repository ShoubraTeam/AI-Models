# ---------------------------------------------------------------------------------------
# The final schema shoud be returned by subagents + post-processing after agents calling
# ---------------------------------------------------------------------------------------


from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated
from schemas import Summary

Score   = Annotated[float, Field(ge = 0.0, le = 1.0)]
Reason  = Annotated[str, Field(min_length = 10, max_length = 50)]


class SubagentResult(BaseModel):
    """
    General Result schema for any sub-agent
    """
    model_config = ConfigDict(
        strict = True,
        validate_assignment = True,
        extra = "forbid"
    )

    score: Score
    accepted: bool
    summary: Summary

    acceptance_reasons: list[Reason] | None = None
    rejection_reasons : list[Reason] | None = None
