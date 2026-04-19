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


class ToolsAlignment(BaseModel):
    job_tools: list[str] = Field(description = "List of the technical tools/frameworks that the client mentioned in their job description")
    proposal_tools: list[str] = Field(description = "List of the technical tools/frameworks that the freelancer mentioned in their proposal")
    confidence_score: float = Field(description = "How much are you (as a model) confident in your analysis & results")
