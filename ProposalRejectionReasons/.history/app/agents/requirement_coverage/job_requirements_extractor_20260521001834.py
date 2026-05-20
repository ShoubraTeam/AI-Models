from agents.BaseAgent import BaseAgent
from schemas..requirement_coveragزchema import ExtractedRequirementsSchema
from prompts.requirement_coverage.job_requirements_extraction_prompt import REQUIREMENT_EXTRACTOR_PROMPT
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

class JobRequirementsExtractor(BaseAgent):
    """العميل الداخلي المسؤول عن استخراج المتطلبات من نص الوظيفة"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        # بننادي على الـ init بتاع الـ BaseAgent لو فيه منطق مشترك
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        
        # بنشغل دالتين التهيئة ورا بعض
        self.init_model(self.model_name, self.temperature)
        self.init_agent()

    def init_model(self, model_name: str, temperature: float = 0.0):
        """تهيئة موديل OpenAI باستخدام الـ API Key المستدعى من البيئة"""
        # الـ API key المفروض مقروء أوتوماتيك من ملف الـ .env عبر الـ BaseAgent أو main.py
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def init_agent(self):
        """ربط الموديل بالـ System Prompt والـ Pydantic Schema"""
        # 1. بنعمل قالب البرومبت ونحدد إن نص الوظيفة هيدخل كـ متغير اسمه job_description
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", REQUIREMENT_EXTRACTOR_PROMPT),
            ("user", "Extract requirements from this job description:\n\n{job_description}")
        ])
        
        # 2. بنجبر الـ LLM يطلع المخرجات مهيكلة حسب الـ Schema بتاعتنا
        self.structured_llm = self.llm.with_structured_output(ExtractedRequirementsSchema)
        
        # 3. بنربط البرومبت مع الموديل المهيكل في Pipeline واحدة
        self.extractor_chain = self.prompt_template | self.structured_llm

    def invoke(self, job_description: str) -> ExtractedRequirementsSchema:
        """تشغيل العميل واستقبال النتيجة كـ Object جاهز"""
        # بنشغل الـ Chain ونباصي لها نص الوظيفة
        response = self.extractor_chain.invoke({"job_description": job_description})
        return response