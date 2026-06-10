# ---------------------------------------------------------------------------------------
# The final schema shoud be returned by subagents + post-processing after agents calling
# ---------------------------------------------------------------------------------------


from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated
from .schema_config import Summary

Score   = Annotated[float, Field(ge = 0.0, le = 1.0)]
Reason  = Annotated[str, Field(min_length = 10, max_length = 100)]


class FinalSubagentResult(BaseModel):
    """
    General Result schema for any sub-agent

    Attbs:
        - score             : the score at the level of task
        - accepted          : if the proposal marked as accepted or not by comparing the score to a threshold related to the task
        - summary           : summary given by the sub-agent highlighting the proposal strenghts & weaknesses.
        - acceptance_reasons: if accepted --> clear short sentences justifying acceptance
        - rejection_reasons : if rejected --> clear short sentences justifying rejection 
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
