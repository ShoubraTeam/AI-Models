# Pydantic Models for IP/OP

from pydantic import BaseModel

class ToolsDetectionIP(BaseModel):
    job_title: str
    job_desc: str

class JobEnhancementIP(BaseModel):
    job_title: str
    job_desc: str
    tools: list = None