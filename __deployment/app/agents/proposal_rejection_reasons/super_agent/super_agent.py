

from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from models.pydantic_schemas import SuperAgentResponse
from prompts import SUPER_AGENT_SYSTEM_PROMPT
from helpers.config import get_settings
from typing import Any
import ast
import json
import re

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


    def _find_failed_generation(self, value: Any) -> str | None:
        if isinstance(value, dict):
            if "failed_generation" in value:
                return value["failed_generation"]

            for item in value.values():
                found = self._find_failed_generation(item)
                if found is not None:
                    return found

        elif isinstance(value, (list, tuple)):
            for item in value:
                found = self._find_failed_generation(item)
                if found is not None:
                    return found

        return None


    def _extract_failed_generation(self, error: Exception) -> str | None:
        for attr_name in ("body", "response", "args"):
            attr_value = getattr(error, attr_name, None)
            found = self._find_failed_generation(attr_value)
            if found is not None:
                return found

        error_text = str(error)
        match = re.search(
            r"['\"]failed_generation['\"]\s*:\s*('(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\")",
            error_text,
            flags=re.DOTALL,
        )
        if not match:
            return None

        try:
            return ast.literal_eval(match.group(1))
        except Exception:
            return match.group(1).strip("'\"")


    def _extract_json_payload(self, raw_generation: str) -> Any:
        raw_generation = raw_generation.strip()

        if raw_generation.startswith("<function="):
            raw_generation = raw_generation.split(">", 1)[1]
            raw_generation = raw_generation.rsplit("</function>", 1)[0]

        start_idx = raw_generation.find("{")
        end_idx = raw_generation.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            raise ValueError("Failed generation did not contain a JSON object.")

        payload = json.loads(raw_generation[start_idx:end_idx + 1])

        if isinstance(payload, dict) and "arguments" in payload:
            payload = payload["arguments"]
            if isinstance(payload, str):
                payload = json.loads(payload)

        return payload


    def _recover_structured_response(self, error: Exception):
        if self.structured_response is None:
            raise error

        failed_generation = self._extract_failed_generation(error)
        if not failed_generation:
            raise error

        payload = self._extract_json_payload(failed_generation)
        return self.structured_response.model_validate(payload)


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
