 # -----------------------------------------------------------------------------------
# Contains the BaseAgent class that works as a blue-print for all agents
# -----------------------------------------------------------------------------------

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class BaseAgent:
    """
    General class for building a LangChain agent

    Attbs:
        model_name (str)
        model_provider (str | None)
        system_prompt (str)
        tools (list)
        response_format: Structured Response Format used for Controlling the Agent output
        **kwargs: keyword arguments for model (temperature - max_tokens - ...)

    Methods:
        __init__  : consructing the Agent Module
        init_model: Initiating the Model (LLM)
        init_agent: Creating a LangChain agent from te model
        invoke    : Quering the agent
        evaluate  : Evaluate the agent [the model, system_prompt, tools specifications, etc...] using the eval_data
    """
    def __init__(self, model_name: str, system_prompt: str, response_format, tools: list = [], model_provider: str = None, **kwargs):
        # CFG
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.tools = tools
        self.response_format = response_format
        self.model_provider = model_provider
        self.kwargs = kwargs

        # init agent
        self.init_agent()
    # -----------------------------------------------------------------------------------
    def init_model(self):
        """"""
        self.model = init_chat_model(
            model_provider = self.model_provider,
            model = self.model_name,
            **self.kwargs
        )
    # -----------------------------------------------------------------------------------
    def init_agent(self):
        self.init_model()



        self.agent = create_agent(
            model = self.model,
            tools = self.tools,
            system_prompt = self.system_prompt,
            response_format = ToolStrategy(self.response_format)
        )
    # -----------------------------------------------------------------------------------
    def invoke(self, query: str, stream = False, return_structured_op_only = False):
        """
        Args:
            query (str)                     :  representing the query
            stream (bool)                   : whether to stream the output [for UX] or not 
            return_structured_op_only (bool): whether to return the whole response or the structured only

        Returns:
            response (str): the model response
        """
        messages = {
            "messages" : [
                    {"role" : "user", "content" : query}
            ]
        }

        if stream:
            pass
        else:
            response = self.agent.invoke(messages)

            if return_structured_op_only:
                return response["structured_response"]
            else:
                return response
    # -----------------------------------------------------------------------------------
    def evaluate(self, eval_data: list):
        """
        Args:
            eval_data: List of JSON objects

        Returns:
            results (dict): a dictionary of the model's results
        """
        pass