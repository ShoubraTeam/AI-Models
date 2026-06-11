# -----------------------------------------------
# Using Groq API as a model provider
# -----------------------------------------------

from typing import Any

from groq import Groq
from groq.types.chat import ChatCompletion


class GroqModelsAPI:
    """
    Thin wrapper around the native Groq chat-completions API.

    Args:
        api_key: Groq API key used to initialize the SDK client.
    """
    def __init__(self, api_key: str) -> None:
        self.client = self.init_client(api_key)

    def init_client(self, api_key: str) -> Groq:
        return Groq(api_key=api_key)

    def generate(
        self,
        model_name: str,
        user_input: str,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        return_whole_response: bool = False,
        **kwargs,
    ) -> ChatCompletion | str:
        """
        Invoke a Groq chat model.

        Args:
            model_name: Groq model id, e.g. ``llama-3.3-70b-versatile``.
            user_input: User message content.
            system_prompt: Optional system message content.
            response_format: Optional native Groq response_format payload.
            return_whole_response: Return the full SDK object when True;
                otherwise return the assistant message content.
            **kwargs: Native Groq generation options such as temperature,
                max_tokens, top_p, timeout, and stop.
        """
        messages = self.create_messages(
            user_input=user_input,
            system_prompt=system_prompt,
        )

        completion = self.client.chat.completions.create(
            messages=messages,
            model=model_name,
            stream=False,
            response_format=response_format,
            **kwargs,
        )

        if return_whole_response:
            return completion

        return completion.choices[0].message.content or ""

    def stream(
        self,
        model_name: str,
        user_input: str,
        response_format: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        """Stream text chunks from a Groq chat model."""
        messages = self.create_messages(
            user_input=user_input,
            system_prompt=system_prompt,
        )

        stream = self.client.chat.completions.create(
            messages=messages,
            model=model_name,
            stream=True,
            response_format=response_format,
            **kwargs,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def create_messages(
        self,
        user_input: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build Groq chat messages, omitting the system message when absent."""
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_input})
        return messages
