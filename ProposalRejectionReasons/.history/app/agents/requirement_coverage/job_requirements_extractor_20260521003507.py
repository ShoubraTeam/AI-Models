from agents.BaseAgent import BaseAgent
from schemas.requirement_coverage.requirement_extraction_schema import ExtractedRequirementsSchema
from prompts.requirement_coverage.job_requirements_extraction_prompt import REQUIREMENT_EXTRACTOR_PROMPT
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

class JobRequirementsExtractor(BaseAgent):
    """العميل الداخلي المسؤول عن استخراج المتطلبات من نص الوظيفة"""
    
    