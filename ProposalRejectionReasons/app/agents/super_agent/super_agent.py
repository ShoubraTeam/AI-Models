
import asyncio

from helpers.config import DEFAULT_MODELS_CFG
from pydantic import BaseModel

from agents.BaseAgent import BaseAgent
from schemas import SuperAgentResponse

from prompts import SUPER_AGENT_SYSTEM_PROMPT


class SuperAgent(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt = SUPER_AGENT_SYSTEM_PROMPT,
        structured_response: type[BaseModel] | None = SuperAgentResponse,
        **kwargs,
    ):
        
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG['super_agent']

        super().__init__(model_name, system_prompt, structured_response, **kwargs)

    # ----------------------------- Modeling -------------------------------- #
    def get_agent(self):
        return super().get_agent()
    
    def invoke(self, job_desc: str, proposal: str, subagents_results: str) -> SuperAgentResponse:
        formatted = self.prepare_ip(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = subagents_results
        )

        return super().invoke(input = formatted)
    
    def ainvoke(self, job_desc: str, proposal: str, subagents_results: str) -> SuperAgentResponse:
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

   
