from agents.BaseAgent import BaseAgent
from schemas.requirement_coverage.requirement_extraction_schema import ExtractedRequirementsSchema
from prompts.requirement_coverage.job_requirements_extraction_prompt import REQUIREMENT_EXTRACTOR_PROMPT
import helpers.config as CFG

class JobRequirementsExtractor(BaseAgent):
    
    def __init__(self, model_name: str = CFG.GROQ_LLAMA_70b, temperature: float = None):
        
        if temperature is None:
            temperature = CFG.MODELS_CFG["requirement_coverage_pipeline"]["job_requirements_extractor_temperature"]
            
        max_tokens = CFG.MODELS_CFG["requirement_coverage_pipeline"]["job_requirements_extractor_max_tokens"]
        
        super().__init__(
            model_name=model_name,
            system_prompt=REQUIREMENT_EXTRACTOR_PROMPT,
            model_provider=CFG.PROVIDER_GROQ, # مقروء من الـ Config
            structured_response=ExtractedRequirementsSchema,
            temperature=temperature,
            max_tokens=max_tokens # ضفناها هنا عشان الـ BaseAgent يستغلها
        )