from agents.BaseAgent import BaseAgent
from schemas.requirement_coverage.requirement_coverage_schema import RequirementCoverageSchema
from prompts.requirement_coverage.job_requirements_matching_prompt import REQUIREMENT_MATCHER_PROMPT
import helpers.config as CFG
from typing import List

class JobRequirementsMatcher(BaseAgent):
    """العميل الداخلي المسؤول عن مطابقة المتطلبات بالبربوزال وحساب السكور"""
    
    # خليت الموديل الافتراضي يقرأ الـ 70b من الكنفج علطول
    def __init__(self, model_name: str = CFG.GROQ_LLAMA_70b, temperature: float = None):
        
        # 2. لو متباصاش temperature من برة، اسحب القيمة من الـ MODELS_CFG
        if temperature is None:
            temperature = CFG.MODELS_CFG["requirement_coverage_pipeline"]["job_requirements_matcher_temperature"]
            
        # 3. سحب الـ max_tokens الخاصة بالـ Matcher من الـ MODELS_CFG
        max_tokens = CFG.MODELS_CFG["requirement_coverage_pipeline"]["job_requirements_matcher_max_tokens"]
        
        # 4. تمرير القيم الديناميكية للـ BaseAgent
        super().__init__(
            model_name=model_name,
            system_prompt=REQUIREMENT_MATCHER_PROMPT,
            model_provider=CFG.PROVIDER_GROQ, # مقروء من الكنفج
            structured_response=RequirementCoverageSchema,
            temperature=temperature,
            max_tokens=max_tokens # تمرير الـ tokens للـ BaseAgent
        )

    def invoke(self, job_requirements: List[str], proposal_text: str) -> RequirementCoverageSchema:
        """تجهيز النصين في متغير واحد واستدعاء الـ invoke بتاعة الـ BaseAgent"""
        formatted_input = f"Job Requirements List:\n{job_requirements}\n\nFreelancer Proposal Text:\n{proposal_text}"
        return super().invoke(input=formatted_input)