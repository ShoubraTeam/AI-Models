 # -----------------------------------------------------------------------------------
# Contains the BaseAgent class that works as a blue-print for all agents
# -----------------------------------------------------------------------------------



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
    def __init__(self, model_name: str, system_prompt: str, tools: list, response_format, model_provider: str = None, **kwargs):
        pass
    # -----------------------------------------------------------------------------------
    def init_model(self):
        """"""
        pass
    # -----------------------------------------------------------------------------------
    def init_agent(self):
        pass
    # -----------------------------------------------------------------------------------
    def invoke(self, query: str, stream = False):
        """
        Args:
            query (str)  :  representing the query
            stream (bool): whether to stream the output [for UX] or not 

        Returns:
            response (str): the model response
        """
        pass
    # -----------------------------------------------------------------------------------
    def evaluate(self, eval_data: list):
        """
        Args:
            eval_data: List of JSON objects

        Returns:
            results (dict): a dictionary of the model's results
        """
        pass