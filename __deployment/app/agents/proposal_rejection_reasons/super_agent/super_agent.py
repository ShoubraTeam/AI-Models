

from models.config.agents_config import PRR_DEFAULT_MODELS_CFG
from pydantic import BaseModel

from agents.proposal_rejection_reasons.base_agent import BaseAgent
from models.schemas import PRR_SuperAgentResponse

from prompts import SUPER_AGENT_SYSTEM_PROMPT


class SuperAgent(BaseAgent):
    def __init__(
        self,
        groq_client,
        model_name: str,
        system_prompt = SUPER_AGENT_SYSTEM_PROMPT,
        structured_response: type[BaseModel] | None = PRR_SuperAgentResponse,
        model_provider: str = "groq",
        **kwargs,
    ):
        
        if "temperature" not in kwargs:
            kwargs = PRR_DEFAULT_MODELS_CFG['super_agent']

        super().__init__(
            groq_client = groq_client,
            model_name = model_name, 
            system_prompt = system_prompt, 
            structured_response = structured_response, 
            model_provider  = model_provider,
            **kwargs
        )

    # ----------------------------- Modeling -------------------------------- #
    
    def invoke(self, job_desc: str, proposal: str, subagents_results: str) -> PRR_SuperAgentResponse:
        formatted = self.prepare_ip(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = subagents_results
        )

        return super().invoke(input = formatted)
    
    def ainvoke(self, job_desc: str, proposal: str, subagents_results: str) -> PRR_SuperAgentResponse:
        formatted = self.prepare_ip(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = subagents_results
        )

        return super().ainvoke(input = formatted)
    


    def prepare_ip(
        self,
        job_desc: str,
        proposal: str,
        subagents_results: str,
    ) -> str:
        return (
            "Job Description:\n"
            f"{job_desc}\n\n"
            "Proposal:\n"
            f"{proposal}\n\n"
            "Sub-agents Results:\n"
            f"{subagents_results}\n"
        )

   
