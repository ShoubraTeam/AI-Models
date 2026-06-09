# --------------------------------------------------------
# A general class used to build agents
# --------------------------------------------------------

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ToolStrategy
from typing import Any
import ast
import json
import re

from helpers.config import get_settings

class BaseAgent:
    """
    General class for building & Invoking agents

    Args:
        model_name     (str): the name of the model wits its provider (provider:model).
        system_prompt  (str): the system prompt used to instuct the model to control its output
        tools (list)        : langchain tools that the agent should use
        structured_response : langchain structured response that the agent should return
        **kwargs            : keyword arguments that should control agent behavior [temperature - max_tokens, ...]
    """
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        
        # setup
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.tools = tools
        self.structured_response = structured_response
        self.kwargs = kwargs

        # creating agent
        self.agent = self.get_agent()
    # -------------------------- Creating Agent ------------------------------
    def get_agent(self):
        model_config = dict(self.kwargs)
        extra_model_kwargs = dict(model_config.pop("model_kwargs", {}) or {})
        structured_output_method = model_config.pop("structured_output_method", "function_calling")

        if "top_p" in model_config:
            extra_model_kwargs["top_p"] = model_config.pop("top_p")

        if extra_model_kwargs:
            model_config["model_kwargs"] = extra_model_kwargs

        model = init_chat_model(
            model   = self.model_name,
            api_key = get_settings().GROQ_API_KEY,
            **model_config
        )

        if self.tools:
            return create_agent(
                model = model,
                tools = self.tools,
                system_prompt = self.system_prompt,
                response_format = ToolStrategy(self.structured_response)
            )
        
        else:
            if self.structured_response:
                return model.with_structured_output(
                    self.structured_response,
                    method=structured_output_method,
                )
            else:
                return model


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


    # -------------------------- invoking -----------------------------
    def invoke(self, input: str, return_structured_op_only: bool = True):
        """
        Calling the agent

        Args:
            input (input)                   : the input goes to the user
            return_structured_op_only (bool): whether to return the structured_op_only or the whole response (for debugging)
        """
        # invoke the agent
        if self.tools:
            try:
                response = self.agent.invoke({
                    "messages" : [
                        {"role" : "user", "content" : input}
                    ]
                })
            except Exception as error:
                return self._recover_structured_response(error)
        
        else:
            messages = [
                {"role" : "system", "content" : self.system_prompt},
                {"role" : "user", "content" : input},
            ]

            try:
                return self.agent.invoke(messages)
            except Exception as error:
                return self._recover_structured_response(error)

        # return desired op
        if return_structured_op_only:
            return response["structured_response"]
        
        return response


    async def ainvoke(self, input: str, return_structured_op_only: bool = True):
        """
        Async version of invoke. Use this when multiple independent model calls
        should be awaited concurrently without wrapping sync calls in threads.
        """
        if self.tools:
            try:
                response = await self.agent.ainvoke({
                    "messages" : [
                        {"role" : "user", "content" : input}
                    ]
                })
            except Exception as error:
                return self._recover_structured_response(error)
        else:
            messages = [
                {"role" : "system", "content" : self.system_prompt},
                {"role" : "user", "content" : input},
            ]

            try:
                return await self.agent.ainvoke(messages)
            except Exception as error:
                return self._recover_structured_response(error)

        if return_structured_op_only:
            return response["structured_response"]

        return response
    
    
    def validate_agent_output(self, agent_output: Any):
        """
        Validate if the agent output as expected or not

        Returns:
            validated (bool)
        """
        pass


    def evaluate(self, eval_data: list):
        pass
