from agents.BaseAgent import BaseAgent
from agents.requirement_coverage.job_requirements_extractor import JobRequirementsExtractor
from agents.requirement_coverage.job_requirements_matcher import JobRequirementsMatcher
from schemas.requirement_coverage.requirement_coverage_schema import RequirementCoverageSchema

class RequirementCoverageAgent(BaseAgent):

    def __init__(self, extractor_model: str = "llama-3.1-8b-instant", matcher_model: str = "llama-3.3-70b-versatile"):
        super().__init__(
            model_name=matcher_model,
            system_prompt="Orchestrator Pipeline",
            model_provider="groq"
        )
        self.extractor_model = extractor_model
        self.matcher_model = matcher_model
        
        self.extractor = JobRequirementsExtractor(model_name=self.extractor_model, temperature=0.0)
        self.matcher = JobRequirementsMatcher(model_name=self.matcher_model, temperature=0.0)

    def invoke(self, job_description: str, proposal_text: str) -> RequirementCoverageSchema:
        """تشغيل الـ Pipeline الأوتوماتيكية بالكامل"""
        # 1. الاستخراج التلقائي (نستخدم input= بناءً على الـ BaseAgent الجديد)
        extraction_result = self.extractor.invoke(input=job_description)
        extracted_reqs = extraction_result.job_requirements
        
        # 2. المطابقة البرمجية وحساب السكور النهائي
        final_evaluation = self.matcher.invoke(
            job_requirements=extracted_reqs,
            proposal_text=proposal_text
        )
        
        return final_evaluation