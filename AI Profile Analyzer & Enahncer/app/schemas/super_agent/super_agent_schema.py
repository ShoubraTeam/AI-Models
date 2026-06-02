from pydantic import BaseModel, Field
from typing import List

class SuperAgentSchema(BaseModel):
    overall_score: float = Field(
        description="The calculated final weighted score for the entire profile (0.0 to 1.0) combining all factors."
    )
    executive_summary: str = Field(
        description="A brilliant, concise 3-4 sentence evaluation summarizing the freelancer's current market standing."
    )
    key_strengths: List[str] = Field(
        description="Top outstanding professional advantages found across their visuals, bio, or skills."
    )
    critical_weaknesses: List[str] = Field(
        description="Top immediate deal-breakers or bottlenecks that are ruining their conversion rate."
    )
    prioritized_action_plan: List[str] = Field(
        description="Top steps roadmap telling the freelancer exactly what to fix first to make more money."
    )