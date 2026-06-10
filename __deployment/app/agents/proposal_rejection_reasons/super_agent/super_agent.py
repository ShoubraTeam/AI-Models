from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from models.pydantic_schemas import SuperAgentResponse
from prompts import SUPER_AGENT_SYSTEM_PROMPT
from helpers.config import get_settings
from agents.proposal_rejection_reasons.structured_output import recover_structured_response

class ProposalRejectionSuperAgent:
    def __init__(
        self,
        model_name         : str,
        system_prompt      : str = SUPER_AGENT_SYSTEM_PROMPT,
        structured_response: type[BaseModel] | None = SuperAgentResponse,
        **kwargs
    ):
        self.model_name          = model_name
        self.system_prompt       = system_prompt
        self.kwargs              = kwargs
        self.structured_response = structured_response
        self.agent = self.get_agent()


    def get_agent(self):
        model_config = dict(self.kwargs)
        extra_model_kwargs = dict(model_config.pop("model_kwargs", {}) or {})
        structured_output_method = model_config.pop("structured_output_method", "function_calling")

        if "top_p" in model_config:
            extra_model_kwargs["top_p"] = model_config.pop("top_p")

        if extra_model_kwargs:
            model_config["model_kwargs"] = extra_model_kwargs

        model = init_chat_model(
            api_key = get_settings().GROQ_API_KEY,
            model = self.model_name,
            **model_config
        )

        if self.structured_response:
            return model.with_structured_output(
                self.structured_response,
                method = structured_output_method,
            )
        else:
            return model


    def _format_messages(self, formatted_input: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": formatted_input},
        ]


    def _extract_response(self, response):
        if self.structured_response:
            return response

        return getattr(response, "content", str(response))


    def _recover_structured_response(self, error: Exception):
        return recover_structured_response(
            error = error,
            structured_response = self.structured_response,
        )


    def format_input(
        self,
        job_desc         : str,
        proposal         : str,
        subagents_results: str
    ) -> str:
        return (
            "<job_description>\n"
            f"{job_desc}\n"
            "</job_description>\n\n"
            "<freelancer_proposal>\n"
            f"{proposal}\n"
            "</freelancer_proposal>\n\n"
            "<subagent_evaluation_report>\n"
            f"{subagents_results}\n"
            "</subagent_evaluation_report>"
        )


    def invoke(
        self,
        job_desc         : str,
        proposal         : str,
        subagents_results: str
    ) -> SuperAgentResponse | str:
        formatted_input = self.format_input(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = subagents_results
        )

        try:
            response = self.agent.invoke(
                self._format_messages(formatted_input = formatted_input)
            )
        except Exception as error:
            return self._recover_structured_response(error)

        return self._extract_response(response = response)


    async def ainvoke(
        self,
        job_desc         : str,
        proposal         : str,
        subagents_results: str
    ) -> SuperAgentResponse | str:
        formatted_input = self.format_input(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = subagents_results
        )

        try:
            response = await self.agent.ainvoke(
                self._format_messages(formatted_input = formatted_input)
            )
        except Exception as error:
            return self._recover_structured_response(error)

        return self._extract_response(response = response)
