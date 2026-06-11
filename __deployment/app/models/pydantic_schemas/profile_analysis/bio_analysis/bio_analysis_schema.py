from pydantic import BaseModel, Field
from typing import List

class BioAnalyzerSchema(BaseModel):
    score: float = Field(
        description="Strict copywriting score from 0.0 to 1.0 evaluating market impact and role alignment."
    )
    analysis: List[str] = Field(
        description="A single flat array of strings containing both pros and cons combined. Do NOT create nested dictionaries or keys like strengths or improvements."
    )