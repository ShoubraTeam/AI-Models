from pydantic import BaseModel, Field
from typing import List

class VisualBrandEvaluationSchema(BaseModel):
    score: float = Field(
        description="Strict score from 0.0 to 1.0 evaluating the overall professional quality and business suitability of the profile image description."
    )
    feedback: List[str] = Field(
        description="Actionable, clear bullet points analyzing lighting, background cleanliness, professional attire, and facial expression."
    )