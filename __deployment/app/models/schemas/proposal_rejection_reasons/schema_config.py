from typing import Annotated
from pydantic import Field, ConfigDict

SUMMARY_MIN_CHAR_LENGTH = 1
SUMMARY_MAX_CHAR_LENGTH = 2000

Summary = Annotated[
    str, 
    Field(
        min_length = SUMMARY_MIN_CHAR_LENGTH, 
        max_length = SUMMARY_MAX_CHAR_LENGTH,
            description = f"Summary of the results found highlighting the strengths and weaknesses. Keep it concise; valid length is {SUMMARY_MIN_CHAR_LENGTH}-{SUMMARY_MAX_CHAR_LENGTH} characters."
    ),
]



model_config = ConfigDict(
    validate_assignment = True,
    extra = "forbid",
    strict = True,
)