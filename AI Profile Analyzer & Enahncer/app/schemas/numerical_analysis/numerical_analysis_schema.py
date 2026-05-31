from pydantic import BaseModel, Field
from typing import List

class NumericalAnalyzerSchema(BaseModel):
    score: float = Field(
        description="Overall financial and credibility score from 0.0 to 1.0 evaluating profile positioning."
    )
    pricing_status: str = Field(
        description="Market positioning status. Strictly choose one of: 'Underpriced', 'Overpriced', or 'Fair Market Value'."
    )
    improvements: List[str] = Field(
        description="Action-oriented numerical recommendations (e.g., 'Increase hourly rate to $30', 'Optimize JSS to cross 90%')."
    )