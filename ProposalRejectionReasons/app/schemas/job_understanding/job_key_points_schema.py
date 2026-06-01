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
        description="Domain-specific keywords from the job description "
                    "excluding tools and technologies — focus on skills, "
                    "methodologies, and domain terms (e.g. 'REST API design', "
                    "'agile', 'data modeling'). Tools are handled separately."
    )
