from agents.BaseAgent import BaseAgent
from schemas.requirement_coverage.requirement_extraction_schema import ExtractedRequirementsSchema
from prompts.requirement_coverage.job_requirements_extraction_prompt import REQUIREMENT_EXTRACTOR_PROMPT
# 1. استدعاء ملف الـ Config
import helpers.config as CFG

class JobRequirementsExtractor(BaseAgent):
    """العميل الداخلي المسؤول عن استخراج المتطلبات من نص الوظيفة"""
    
    # خفضنا الـ Defaults لتُقرأ من الـ Config مباشرة
    def __init__(self, model_name: str = CFG.GROQ_LLAMA_70b, temperature: float = None):
        
        # 2. لو متباصاش temperature من برة، اسحب القيمة الافتراضية من الـ MODELS_CFG
        if temperature is None:
            temperature = CFG.MODELS_CFG["requirement_coverage_pipeline"]["job_requirements_extractor_temperature"]
            
        # 3. سحب الـ max_tokens من الـ MODELS_CFG
        max_tokens = CFG.MODELS_CFG["requirement_coverage_pipeline"]["job_requirements_extractor_max_tokens"]
        
        # 4. تمرير كل القيم الديناميكية للـ BaseAgent
        super().__init__(
            model_name=model_name,
            system_prompt=REQUIREMENT_EXTRACTOR_PROMPT,
            model_provider=CFG.PROVIDER_GROQ, # مقروء من الـ Config
            structured_response=ExtractedRequirementsSchema,
            temperature=temperature,
            max_tokens=max_tokens # ضفناها هنا عشان الـ BaseAgent يستغلها
        )