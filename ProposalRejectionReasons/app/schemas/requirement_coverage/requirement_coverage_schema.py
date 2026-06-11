from pydantic import BaseModel, Field
from typing import List

from ..schema_config import Summary, model_config


class RequirementCoverageSchema(BaseModel):
    """Coverage judgment for a proposal against extracted job requirements."""

    model_config = model_config

    details: str = Field(
        description=(
            "Clear justification explaining why each relevant requirement ID "
            "was considered covered, missed, or violated."
        )
    )
    requirements_covered_ids: List[str] = Field(
        description=(
            "Exact original IDs from the input requirements that the proposal "
            "satisfies or respects. Return an empty list when none are covered."
        )
    )
    missing_requirements_ids: List[str] = Field(
        description=(
            "Exact original IDs from the input requirements that the proposal "
            "misses, does not address, or violates. Return an empty list when none are missing."
        )
    )

    summary: Summary
