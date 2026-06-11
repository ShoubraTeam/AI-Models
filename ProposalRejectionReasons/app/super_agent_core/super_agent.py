import os
import json
import asyncio

from typing import Any
from pydantic import BaseModel

from agents.BaseAgent import NativeStructuredOutputError
from schemas import SuperAgentResponse

from groq_core import GroqModelsAPI, get_response_format
from prompts import SUPER_AGENT_SYSTEM_PROMPT


class ProposalRejectionSuperAgent:
    def __init__(
        self,
        model_name: str,
        system_prompt=SUPER_AGENT_SYSTEM_PROMPT,
        structured_response: type[BaseModel] | None = SuperAgentResponse,
        **kwargs,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.structured_response = structured_response
        self.kwargs = self._normalize_kwargs(kwargs)

        self.schema_name = structured_response.__name__ if structured_response else None
        self.prompt_name: str | None = None
        self.agent = self.get_agent()

    @staticmethod
    def _normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        model_kwargs = dict(kwargs.pop("model_kwargs", {}) or {})
        for removed_key in ("structured_output_method", "max_retries", "model_provider"):
            kwargs.pop(removed_key, None)
        model_kwargs.update(kwargs)
        return model_kwargs

    @staticmethod
    def _get_api_key() -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required to use the Groq native super-agent.")
        return api_key

    def get_agent(self) -> GroqModelsAPI:
        return GroqModelsAPI(api_key=self._get_api_key())

    def _get_response_format(self) -> dict[str, Any] | None:
        if self.structured_response is None:
            return None
        return get_response_format(
            model_name=self.model_name,
            schema=self.structured_response,
            schema_name=self.schema_name or "super_agent_response",
        )

    def _resolve_system_prompt(self, response_format_type: str) -> str | None:
        if callable(self.system_prompt):
            resolved_prompt = self.system_prompt(response_format_type)
            if isinstance(resolved_prompt, tuple):
                self.prompt_name = resolved_prompt[0]
                return resolved_prompt[1]
            self.prompt_name = getattr(self.system_prompt, "__name__", None)
            return resolved_prompt
        self.prompt_name = None
        return self.system_prompt

    @staticmethod
    def _extract_json_text(raw_output: str) -> str:
        stripped = raw_output.strip()
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

        start_idx = stripped.find("{")
        end_idx = stripped.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            raise ValueError("No JSON object found in model output.")
        return stripped[start_idx:end_idx + 1]

    def _extract_response(self, raw_response: Any):
        if self.structured_response is None:
            return raw_response

        try:
            if isinstance(raw_response, str):
                raw_response = json.loads(self._extract_json_text(raw_response))
            return self.structured_response.model_validate(raw_response)
        except Exception as error:
            raise NativeStructuredOutputError(
                message=f"Failed to validate Groq native output against {self.schema_name}.",
                raw_output=raw_response,
                schema_name=self.schema_name,
            ) from error

    def format_input(
        self,
        job_desc: str,
        proposal: str,
        subagents_results: str,
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
        job_desc: str,
        proposal: str,
        subagents_results: str,
    ) -> SuperAgentResponse | str:
        formatted_input = self.format_input(
            job_desc=job_desc,
            proposal=proposal,
            subagents_results=subagents_results,
        )
        response_format = self._get_response_format()
        response_format_type = response_format["type"] if response_format else "text"
        system_prompt = self._resolve_system_prompt(response_format_type)

        raw_response = self.agent.generate(
            model_name=self.model_name,
            user_input=formatted_input,
            system_prompt=system_prompt,
            response_format=response_format,
            **self.kwargs,
        )
        
        return self._extract_response(raw_response)

    async def ainvoke(
        self,
        job_desc: str,
        proposal: str,
        subagents_results: str,
    ) -> SuperAgentResponse | str:
        return await asyncio.to_thread(
            self.invoke,
            job_desc,
            proposal,
            subagents_results,
        )
