# --------------------------------------
# Shared schema type aliases and settings
# --------------------------------------

from typing import Annotated
from pydantic import Field, ConfigDict

SUMMARY_MIN_CHAR_LENGTH = 10
SUMMARY_MAX_CHAR_LENGTH = 250

Summary = Annotated[
    str,
    Field(
        description=(
            "Concise summary of the evaluation result, including the most "
            f"important strengths and weaknesses when relevant. Target length: "
            f"{SUMMARY_MIN_CHAR_LENGTH}-{SUMMARY_MAX_CHAR_LENGTH} characters."
        )
    ),
]


model_config = ConfigDict(
    validate_assignment = True,
    extra = "forbid",
    strict = True,
)