from agents.BaseAgent import BaseAgent
from agents.requirement_coverage.job_requirements_extractor import JobRequirementsExtractor
from agents.requirement_coverage.job_requirements_matcher import JobRequirementsMatcher
from schemas.requirement_coverage.requirement_coverage_schema import RequirementCoverageSchema
# 1. استدعاء ملف الـ Config المركزي
import helpers.config as CFG

class RequirementCoverageAgent(BaseAgent):

    # 2. ربط الموديلات الافتراضية بمتغيرات الـ Config واستخدام الـ 70b لمنع الـ Errors
    def __init__(self, extractor_model: str = CFG.GROQ_LLAMA_70b, matcher_model: str = CFG.GROQ_LLAMA_70b):
        super().__init__(
            model_name=matcher_model,
            system_prompt="Orchestrator Pipeline",
            model_provider=CFG.PROVIDER_GROQ # 3. قراءة الـ Provider من الـ Config
        )
        self.extractor_model = extractor_model
        self.matcher_model = matcher_model
        
        # 4. استدعاء الميكرو-إيجنتس الداخليين (بياخدوا الـ Temperature والـ Tokens تلقائياً من الـ Config جواهم)
        self.extractor = JobRequirementsExtractor(model_name=self.extractor_model)
        self.matcher = JobRequirementsMatcher(model_name=self.matcher_model)

    def invoke(self, job_description: str, proposal_text: str) -> RequirementCoverageSchema:
        """تشغيل الـ Pipeline بالكامل أوتوماتيك"""
        
        # الخطوة الأولى: استخراج المتطلبات
        extraction_result = self.extractor.invoke(input=job_description)
        extracted_reqs = extraction_result.job_requirements
        
        # الخطوة الثانية: المطابقة وحساب السكور
        final_evaluation = self.matcher.invoke(
            job_requirements=extracted_reqs,
            proposal_text=proposal_text
        )
        
        return final_evaluation