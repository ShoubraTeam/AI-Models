# -----------------------------------------------------------------
# Implementing an Agent to extract tools from the job description
# -----------------------------------------------------------------


from ..BaseAgent import BaseAgent
from schemas import JobTool, JobToolResponse
# from typing import get_args
from helpers.config import DEFAULT_MODELS_CFG

class JobToolsExtractor(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG['job_tools_extractor']

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)
    
    def get_agent(self):
        return super().get_agent()
    
    def invoke(self, input, return_structured_op_only = True):
        return super().invoke(input, return_structured_op_only)
    # -------------------------------------------------------------------------------
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)
    
    # ---------------------------- Evaluation ----------------------------------
    def compare_true_tools_to_preds(self, true_tools: list[dict], pred_tools: JobToolResponse):
        """
        Args:
            true_tools: list of {tool, necessity_level} dict
            pred_tools: list of {tool, necessity_level} data schema
        """
        # calc tool scores
        true_tool_names = set([tool["tool_name"] for tool in true_tools])
        pred_tool_names = set([tool.tool_name for tool in pred_tools])

        tools_acc = true_tool_names.intersection(pred_tool_names) / len(true_tools)
        # tools_precision = 
    



    
    def evaluate(self, job_data: list[dict], return_agent_response: bool = False):
        """
        Evaluating the agent on the given list of data

        Args:
            job_data: list of job data samples. Each sample contains ["desc" & "tools"] (job desc & job tools)
        """

        tools_precision = 0.0
        tools_recall = 0.0
        tools_accuracy = 0.0
        for sample in job_data:
            acc = 0.0
            count = 0

            # get data
            job_desc = sample["desc"]
            true_job_tools = sample["tools"]

            # extract tools using agents
            pred_job_tools = self.invoke(input = job_desc)

            # compare pred to true
            


        

        
