from pydantic import BaseModel, Field

# ----------------------------- Modifications -------------------------------
# --> Remove confidence_score
# ---------------------------------------------------------------------------
from ..schema_config import Summary
class LanguageClarityEvalSchema(BaseModel):
    """
    Output of LanguageClarityEvaluator.

    ONLY contains decisions that require LLM reasoning.
    Text metrics (word count, sentence length) are calculated
    separately in processing using normal code.
    """

    is_clear: bool = Field(
        description="Whether the proposal is easy to understand. "
                    "False if sentences are confusing, overly complex, or hard to follow."
    )
    
    is_professional: bool = Field(
        description="Whether the tone is professional and appropriate for a client. "
                    "False if the tone is too casual, rude, or unprofessional."
    )
    has_misleading_phrasing: bool = Field(
        description="Whether the proposal contains vague or misleading statements "
                    "such as 'I can do everything', 'guaranteed results', or empty promises."
    )
    summary: Summary
    
    confidence_score: float = Field(
        description="How confident the agent is in its evaluation. Between 0.0 and 1.0."
    )
