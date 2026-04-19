# -----------------------------------------------------------------------------------
# Contains the Tool Alignment Agent
# should inherit the BaseAgent module
# -----------------------------------------------------------------------------------


from agents.base_agent import BaseAgent


class ToolAlignmentAgent(BaseAgent):
    # Please, read the BaseAgent class for more info!!!!!!!!!
    def __init__(self, model_name, system_prompt, tools, response_format, model_provider = None, **kwargs):
        super().__init__(model_name, system_prompt, tools, response_format, model_provider, **kwargs)
    # -----------------------------------------------------------------------------------
    def init_model(self):
        pass
    # -----------------------------------------------------------------------------------
    def init_agent(self):
        pass
    # -----------------------------------------------------------------------------------
    def invoke(self, query, stream = False):
        pass
    # -----------------------------------------------------------------------------------
    def evaluate(self, eval_data):
        pass