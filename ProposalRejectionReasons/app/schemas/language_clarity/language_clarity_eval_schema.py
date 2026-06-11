from pydantic import BaseModel, Field

from ..schema_config import Summary, model_config


class LanguageClarityEvalSchema(BaseModel):
    """
    Proposal language-quality evaluation.

    This schema contains only LLM judgments about clarity, professionalism,
    vague phrasing, and a concise explanation. Text statistics are handled by
    the processing layer.
    """
    model_config = model_config

    is_clear: bool = Field(
        description=(
            "Whether the proposal is easy to understand. False when wording, "
            "grammar, structure, or sentence flow makes the proposal hard to follow."
        )
    )

    is_professional: bool = Field(
        description=(
            "Whether the tone is appropriate for a client-facing freelance "
            "proposal. False for careless, rude, overly casual, or unprofessional wording."
        )
    )
    has_misleading_phrasing: bool = Field(
        description=(
            "Whether the proposal contains vague, exaggerated, unsupported, or "
            "misleading claims such as broad guarantees or empty promises."
        )
    )
    summary: Summary

    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the language-quality evaluation from 0.0 to 1.0.",
    )
