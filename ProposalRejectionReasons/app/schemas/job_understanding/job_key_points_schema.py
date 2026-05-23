from pydantic import BaseModel, Field
from typing import List


class JobKeyPointsSchema(BaseModel):
    """
    Output of JobKeyPointsExtractor.
    Extracts structured key points from the job description.
    Used downstream by the evaluator and by metric calculations.
    """
    core_problem: str = Field(
        description="The main problem or goal the client wants to solve."
    )

    required_deliverables: List[str] = Field(
        description="List of concrete deliverables or outcomes the client expects."
    )
    
    key_keywords: List[str] = Field(
        description="Important domain-specific keywords extracted from the job description "
                    "(tools, skills, technologies, methodologies)."
    )
