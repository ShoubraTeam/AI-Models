# -----------------------------------------------------------------------------------
# Pydantic Structured Response Classes for the Agents
# -----------------------------------------------------------------------------------

from pydantic import BaseModel, Field


# Job Understanding: example
class JobUnderstanding(BaseModel):
    score: float = Field(
        description = "score indicates ....",  # وصف مناسب عشان الموديل يتعرف عليه
        le = 1.0,  # اصغر من 1
        ge = 0.0  # اكبر من 0
    )

    # details: str = Field 


    