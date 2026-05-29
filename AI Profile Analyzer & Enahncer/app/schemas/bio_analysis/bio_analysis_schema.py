from pydantic import BaseModel, Field
from typing import List

class BioAnalyzerSchema(BaseModel):
    score: float = Field(
        description="Strict copywriting score from 0.0 to 1.0 evaluating marketing impact, role alignment, and engagement."
    )
    strengths: List[str] = Field(
        description="Key positive aspects found in the bio (e.g., strong hook, clear niche definition)."
    )
    improvements: List[str] = Field( 
        description="Action-oriented bullet points detailing specific issues like weak value proposition, grammatical errors, or fluff."
    )