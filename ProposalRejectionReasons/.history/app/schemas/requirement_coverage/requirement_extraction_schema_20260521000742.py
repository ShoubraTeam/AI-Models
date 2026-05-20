from pydantic import BaseModel, Field
from typing import List

class ExtractedRequirementsSchema(BaseModel):
    job_requirements: List[str] = Field(
        description="A clean, atomic list of core functional requirements or features requested in the job description. Exclude specific tools or coding languages."
    )