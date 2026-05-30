from pydantic import BaseModel, Field


# ------------- Modifications ------------
# --> Removing confidence_score
# --> Adding detecting proposal keywords using agent
# ---------------------------------------

class JobUnderstandingEvalSchema(BaseModel):
    """
    Output of JobUnderstandingEvaluator.

    ONLY contains decisions that require LLM reasoning.
    Keyword matching, coverage scores, and similarity metrics
    are calculated separately in processing using normal code.
    """

    problem_identified: bool = Field(
        description="Whether the freelancer clearly identified the core problem stated in the job."
    )

    solution_proposed: bool = Field(
        description="Whether the freelancer proposed a concrete and relevant solution."
    )
    
    practical_steps_mentioned: bool = Field(
        description="Whether the freelancer mentioned practical or actionable steps."
    )

    summary: str = Field(
        description="A short 1-2 sentence explanation of the evaluation result."
    )
    
    confidence_score: float = Field(
        description="How confident the agent is in its evaluation. Between 0.0 and 1.0."
    )
