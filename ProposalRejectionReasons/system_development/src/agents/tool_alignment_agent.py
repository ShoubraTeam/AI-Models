# -----------------------------------------------------------------------------------
# Contains the Tool Alignment Agent
# should inherit the BaseAgent module
# -----------------------------------------------------------------------------------


from src.agents.base_agent import BaseAgent


class ToolAlignmentAgent(BaseAgent):
    # Please, read the BaseAgent class for more info!!!!!!!!!
    def __init__(self, model_name: str, system_prompt: str, response_format, tools: list = [], model_provider: str = None, **kwargs):
        super().__init__(model_name, system_prompt, response_format, tools, model_provider, **kwargs)
    # -----------------------------------------------------------------------------------
    def init_model(self):
        return super().init_model()
    # -----------------------------------------------------------------------------------
    def init_agent(self):
        return super().init_agent()
    # -----------------------------------------------------------------------------------
    def invoke(self, query, stream = False, return_structured_op_only = False):
        return super().invoke(query, stream, return_structured_op_only)
    # -----------------------------------------------------------------------------------
    def evaluate(self, eval_data):
        pass