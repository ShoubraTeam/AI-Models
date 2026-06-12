# --------------------------------------------------------
# A general class used to build Groq-native agents
# --------------------------------------------------------

import os
import json
import asyncio

from typing import Any
from pydantic import BaseModel

from .groq_core import GroqModelsAPI, get_response_format


class NativeStructuredOutputError(RuntimeError):
    """Raised when Groq returns content that cannot validate against a schema."""

    def __init__(self, message: str, raw_output: Any = None, schema_name: str | None = None):
        super().__init__(message)
        self.raw_output = raw_output
        self.schema_name = schema_name


class BaseAgent:
    """
    Base class for Groq-native structured-output agents.

    Args:
        model_name         : Groq model name. Also, handle the prefix (groq:). 
        system_prompt      : the system prompt. Can be either:
                                - static text
                                - callable receives response format type
        structured_response: Pydantic model class used to validate the returned JSON.
        **kwargs           : generation configurations [temperature - max_tokens - ...]
    """

    def __init__(
        self,
        groq_client: GroqModelsAPI,
        model_name: str,
        system_prompt: str | Any,
        structured_response: type[BaseModel],
        **kwargs,
    ):
        # setup
        self.model_name          = model_name
        self.system_prompt       = system_prompt
        self.structured_response = structured_response

        self.kwargs = self._normalize_kwargs(kwargs)
        self.schema_name = self._get_schema_name(structured_response)
        self.prompt_name: str | None = None
        self.agent = groq_client


    @staticmethod
    def _get_api_key() -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required to use Groq native agents.")
        return api_key
    

    @staticmethod
    def _normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        model_kwargs = dict(kwargs.pop("model_kwargs", {}) or {})

        for removed_key in (
            "model_provider",
            "structured_output_method",
            "max_retries",
            "tools",
            "tool_choice",
        ):
            kwargs.pop(removed_key, None)

        model_kwargs.update(kwargs)
        return model_kwargs

    # ---------------------------- Schema & Prompt -------------------------------
    @staticmethod
    def _is_schema_class(candidate: Any) -> bool:
        try:
            return isinstance(candidate, type) and issubclass(candidate, BaseModel)
        except TypeError:
            return False


    @staticmethod
    def _get_schema_name(schema: type[BaseModel] | None) -> str | None:
        if schema is None:
            return None
        return schema.__name__


    def _get_response_format(self) -> dict[str, Any] | None:
        if self.structured_response is None:
            return None

        return get_response_format(
            model_name  = self.model_name,
            schema      = self.structured_response,
            schema_name = self.schema_name or "structured_response",
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

    # ---------------------- Parsing Model OP ----------------------- #
    @staticmethod
    def _extract_json_text(raw_output: str) -> str:
        stripped = raw_output.strip()

        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()

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



    @staticmethod
    def _unwrap_arguments(payload: Any) -> Any:
        if isinstance(payload, dict) and "arguments" in payload:
            payload = payload["arguments"]
            if isinstance(payload, str):
                payload = json.loads(payload)
        return payload


    def _validate_structured_response(self, raw_output: Any):
        # no structured op
        if self.structured_response is None:
            return raw_output

        # no need for parsing
        if isinstance(raw_output, self.structured_response):
            return raw_output

        # parse
        try:
            if isinstance(raw_output, str):
                json_text = self._extract_json_text(raw_output)
                payload = json.loads(json_text)
            else:
                payload = raw_output

            payload = self._unwrap_arguments(payload)
            return self.structured_response.model_validate(payload)
        
        except Exception as error:
            raise NativeStructuredOutputError(
                message = (
                    f"Failed to validate Groq native output against "
                    f"{self.schema_name or 'structured response schema'}."
                ),
                raw_output = raw_output,
                schema_name = self.schema_name,
            ) from error
        

    # -------------------------- Modeling ---------------------- #


    def invoke(self, input: str, *_, **__):
        response_format = self._get_response_format()
        response_format_type = response_format["type"] if response_format else "text"

        system_prompt = self._resolve_system_prompt(response_format_type)

        raw_response = self.agent.generate(
            model_name      = self.model_name,
            user_input      = input,
            system_prompt   = system_prompt,
            response_format = response_format,
            **self.kwargs,
        )

        return self._validate_structured_response(raw_response)


    async def ainvoke(self, input: str, *_, **__):
        return await asyncio.to_thread(
            BaseAgent.invoke,
            self,
            input,
        )

    def validate_agent_output(self, agent_output: Any):
        """Validate or inspect an agent output. Subclasses may override."""
        pass

    def evaluate(self, eval_data: list):
        pass
