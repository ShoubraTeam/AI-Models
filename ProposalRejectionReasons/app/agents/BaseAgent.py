# --------------------------------------------------------
# A general class used to build agents
# --------------------------------------------------------

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ToolStrategy

class BaseAgent:
    """
    General class for building & Invoking agents

    Args:
        model_name     (str): the name of the model used.
        model_provider (str): the provider name. e.g Groq, OpenAI, ...
        system_prompt  (str): the system prompt used to instuct the model to control its output
        tools (list)        : langchain tools that the agent should use
        structured_response : langchain structured response that the agent should return
        **kwargs            : keyword arguments that should control agent behavior [temperature - max_tokens, ...]
    """
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        model_provider: str = "groq",
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        
        # setup
        self.model_name = model_name
        self.model_provider = model_provider
        self.system_prompt = system_prompt
        self.tools = tools
        self.structured_response = structured_response
        self.kwargs = kwargs

        # creating agent
        self.agent = self.get_agent()
    # -------------------------- Creating Agent ------------------------------
    def get_agent(self):
        model = init_chat_model(
            model = self.model_name,
            model_provider = self.model_provider
            **self.kwargs
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
                return model.with_structured_output(self.structured_response)
            else:
                return model
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
            response = self.agent.invoke({
                "messages" : [
                    {"role" : "user", "content" : input}
                ]
            })
        
        else:
            messages = [
                {"role" : "system", "content" : self.system_prompt},
                {"role" : "user", "content" : input},
            ]

            response = self.agent.invoke(messages)

        # return desired op
        if return_structured_op_only:
            return response["structured_response"]
        
        return response
    # -------------------------------------------------------