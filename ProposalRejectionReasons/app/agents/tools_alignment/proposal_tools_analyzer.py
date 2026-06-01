# -----------------------------------------------------------------
# Implementing an Agent to extract tools from the job description
# -----------------------------------------------------------------


from ..BaseAgent import BaseAgent
from helpers.config import DEFAULT_MODELS_CFG
from processing.tool_alignment_processing import format_ip_for_proposal_tools_analyzer

class ProposalToolsAnalyzer(BaseAgent):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        
        if "temperature" not in kwargs:
            kwargs = DEFAULT_MODELS_CFG["proposal_tools_analyzer"]

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)
    
    def get_agent(self):
        return super().get_agent()
    
    def invoke(self, input, return_structured_op_only = True):
        return super().invoke(input, return_structured_op_only)

    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)
    
    # ---------------------------------------------------- Evaluation -----------------------------------------------------
    # stopped here
    def calc_tool_names_metrics(self, true_tool_names: list[str], pred_tool_names: list[str]) -> dict[str, float]:
        """
        Args:
            true_tools: list of true tool names 
            pred_tools: list of pred tool names
        """
        # calc matches
        matched_true = set()
        matched_pred = set()

        for pred_idx, pred_name in enumerate(pred_tool_names):
            for true_idx, true_name in enumerate(true_tool_names):
                if true_idx not in matched_true and self.is_match(true_name, pred_name):
                    matched_true.add(true_idx)
                    matched_pred.add(pred_idx)
                    break
        
        # calc scores
        TP = len(matched_pred)                          
        FP = len(pred_tool_names) - len(matched_pred)
        FN = len(true_tool_names) - len(matched_true)

        accuracy  = TP / (TP + FP + FN) if (TP + FP + FN) else 0
        precision = TP / (TP + FP)      if (TP + FP)      else 0
        recall    = TP / (TP + FN)      if (TP + FN)      else 0

        return {
            "accuracy"      : accuracy,
            "precision"     : precision,
            "recall"        : recall,
        }
    
    
    
    
    def evaluate(self, eval_data: list[dict]) -> dict[str, list]:
        """
        Evaluating the agent on the given list of data

        Args:
            job_data: list of job data samples. Each sample contains ["desc" & "tools"] (job desc & job tools)
        """
        
        for idx, sample in enumerate(eval_data, start = 1):
            print(f">> Evaluating on sample #{idx}")
            print()

            # job data
            job_desc = sample['job']["desc"]
            job_tools = sample['job']["tools"]

            # proposals
            proposals = sample["proposals"]
            n_proposals = len(proposals)

            for p in proposals:
                proposal = p["proposal"]
                proposal_tools = p["tools"]

                formatted_agent_ip = format_ip_for_proposal_tools_analyzer(
                    job_tools = job_tools,
                    proposal = proposal
                )

                agent_response = self.invoke(input = formatted_agent_ip)
                tool_reviews = agent_response.tool_reviews


        
