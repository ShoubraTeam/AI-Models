# --------------------------------------
# Schema Configuraitons
# --> Types
# --> ...
# --------------------------------------

from typing import Annotated
from pydantic import Field

SUMMARY_MIN_CHAR_LENGTH = 100
SUMMARY_MAX_CHAR_LENGTH = 500

Summary = Annotated[
    str, 
    Field(
        min_length = SUMMARY_MIN_CHAR_LENGTH, 
        max_length = SUMMARY_MAX_CHAR_LENGTH,
        description = f"Summary of the results found highlighting the strengths and weaknesses. It should have a minimum length of {SUMMARY_MIN_CHAR_LENGTH} characters and maximum length of {SUMMARY_MAX_CHAR_LENGTH} characters."
    ),
]