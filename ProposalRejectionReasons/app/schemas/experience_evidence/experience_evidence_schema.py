from pydantic import BaseModel, Field
from typing import List
from schemas import Summary

###
# Modifications:
# --> if has_experience_evidence
# ------> return list of projects with [summary & relevance score] 

# --> if has_experience_evidence == False
# ------> return empty list

# --> in both cases -> return has_experience_evidence flag.

# --> discussing putting links in proposal!!
### 
class ExtractedProject(BaseModel):
    project_title: str = Field(
        description="The name or type of the past project mentioned by the freelancer."
    )

    project_description: str = Field(
        description="Brief details of what the freelancer actually did in this past project."
    )
    
    tools_used: List[str] = Field(
        description="List of technologies, languages, or tools explicitly mentioned within this specific past project context."
    )

    relevance_analysis: str = Field(
        description="A direct technical analysis explaining how this past project relates to the current Job Description.")

class ExperienceEvidenceSchema(BaseModel):
    has_experience_evidence: bool = Field(
        description="True ONLY if the freelancer explicitly mentions specific past projects or hands-on built solutions. False if they only provide generic claims of years of experience without context."
    )

    extracted_projects: List[ExtractedProject] = Field(
        description="List of all validated past projects extracted from the proposal text. Must be empty if has_experience_evidence is False."
    )

    summary: Summary