from agents.BaseAgent import BaseAgent
from schemas.reqrequirement_coverage_schema import RequirementCoverageSchema
from prompts.requirement_coverage.job_requirements_matching_prompt import REQUIREMENT_MATCHER_PROMPT
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import List

class JobRequirementsMatcher(BaseAgent):
    """العميل الداخلي المسؤول عن مطابقة المتطلبات بالبربوزال وحساب السكور"""
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        
        self.init_model(self.model_name, self.temperature)
        self.init_agent()

    def init_model(self, model_name: str, temperature: float = 0.0):
        """تهيئة موديل Groq"""
        self.llm = ChatGroq(model=model_name, temperature=temperature)

    def init_agent(self):
        """ربط الموديل بالـ Matcher Prompt والـ القالب النهائي"""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", REQUIREMENT_MATCHER_PROMPT),
            ("user", "Job Requirements List:\n{job_requirements}\n\nFreelancer Proposal Text:\n{proposal_text}")
        ])
        
        # بنجبر الـ LLM يطلع المخرجات مهيكلة حسب الـ RequirementCoverageSchema
        self.structured_llm = self.llm.with_structured_output(RequirementCoverageSchema)
        
        # ربط البرومبت بالموديل المهيكل
        self.matcher_chain = self.prompt_template | self.structured_llm

    def invoke(self, job_requirements: List[str], proposal_text: str) -> RequirementCoverageSchema:
        """تشغيل عملية المطابقة وحساب السكور"""
        # بنباصي المتغيرين للـ Chain
        response = self.matcher_chain.invoke({
            "job_requirements": job_requirements,
            "proposal_text": proposal_text
        })
        return response