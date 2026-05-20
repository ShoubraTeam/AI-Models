from agents.BaseAgent import BaseAgent
from agents.requirement_coverage.job_requirements_extractor import JobRequirementsExtractor
from agents.requirement_coverage.job_requirements_matcher import JobRequirementsMatcher
from schemas..requirement_coverage_schema import RequirementCoverageSchema

class RequirementCoverageAgent(BaseAgent):
    """
    الـ Agent الرئيسي الخاص بتقييم تغطية المتطلبات.
    يدير داخلياً الـ Extractor والـ Matcher لتشغيل الـ Pipeline بالكامل أوتوماتيك.
    """
    
    def __init__(self, extractor_model: str = "llama-3.1-8b-instant", matcher_model: str = "llama-3.3-70b-versatile"):
        """
        بنباصي موديل خفيف وسريع للاستخراج (زي Llama 8B)
        وموديل قوي وذكي للمطابقة والحساب (زي Llama 70B) لرفع الجودة وتقليل التكلفة
        """
        super().__init__()
        self.extractor_model = extractor_model
        self.matcher_model = matcher_model
        
        # تشغيل دالة التهيئة
        self.init_model()
        self.init_agent()

    def init_model(self):
        """تهيئة الميكرو-إيجنتس الداخليين بالكامل"""
        self.extractor = JobRequirementsExtractor(model_name=self.extractor_model, temperature=0.0)
        self.matcher = JobRequirementsMatcher(model_name=self.matcher_model, temperature=0.0)

    def init_agent(self):
        """الـ Agents الداخليين بيهيئوا نفسهم تلقائياً، فلا نحتاج منطق إضافي هنا"""
        pass

    def invoke(self, job_description: str, proposal_text: str) -> RequirementCoverageSchema:
        """
        الـ Pipeline الأوتوماتيكية بالكامل:
        1. العميل يدخل نص الوظيفة عمياني -> يتم استخراج الـ Requirements كـ List.
        2. نمرر الـ List مع الـ Proposal للـ Matcher -> يتم حساب السكور والنواقص.
        """
        # الخطوة 1: الاستخراج التلقائي للمتطلبات
        extraction_result = self.extractor.invoke(job_description=job_description)
        extracted_reqs = extraction_result.job_requirements
        
        # الخطوة 2: المطابقة البرمجية وحساب السكور النهائي
        final_evaluation = self.matcher.invoke(
            job_requirements=extracted_reqs,
            proposal_text=proposal_text
        )
        
        # إرجاع الكائن النهائي المنظم للـ SuperAgent
        return final_evaluation