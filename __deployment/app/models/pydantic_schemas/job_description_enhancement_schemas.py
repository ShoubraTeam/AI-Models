

from pydantic import BaseModel

class ToolsDetectionIP(BaseModel):
    job_description : str

class ToolsRecommendationIP(BaseModel):
    job_title       : str
    job_description : str

class JobEnhancementIP(BaseModel):
    job_title      : str
    job_description: str
    tools          : list = None