from agents.BaseAgent import BaseAgent
from schemas.requirement_coverage.requirement_extraction_schema import ExtractedRequirementsSchema
from prompts.requirement_coverage.job_requirements_extraction_prompt import REQUIREMENT_EXTRACTOR_PROMPT
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

class JobRequirementsExtractor(BaseAgent):
    """العميل الداخلي المسؤول عن استخراج المتطلبات من نص الوظيفة"""
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.0):
        # بنباصي كل المتغيرات للـ BaseAgent وهو هيعمل الـ initialization والـ structure أوتوماتيك
        super().__init__(
            model_name=model_name,
            system_prompt=REQUIREMENT_EXTRACTOR_PROMPT,
            model_provider="groq",
            structured_response=ExtractedRequirementsSchema,
            temperature=temperature
        )