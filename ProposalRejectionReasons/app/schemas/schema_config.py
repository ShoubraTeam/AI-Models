# --------------------------------------
# Schema Configuraitons
# --> Types
# --> ...
# --------------------------------------

from typing import Annotated
from pydantic import Field

SUMMARY_MIN_LENGTH = 50
SUMMARY_MAX_LENGTH = 150

Summary = Annotated[
    str, 
    Field(
        min_length = SUMMARY_MIN_LENGTH, 
        max_length = SUMMARY_MAX_LENGTH,
        description = f"Summary of the results found. It should have a minimum length of {SUMMARY_MIN_LENGTH} words and maximum length of {SUMMARY_MAX_LENGTH} words."
    ),
]