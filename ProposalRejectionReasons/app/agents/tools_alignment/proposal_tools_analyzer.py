# -----------------------------------------------------------------
# Implementing an Agent to extract tools from the job description
# -----------------------------------------------------------------


from ..BaseAgent import BaseAgent
from helpers.config import DEFAULT_MODELS_CFG
from processing.tool_alignment_processing import format_ip_for_proposal_tools_analyzer
from schemas import ProposalToolReview, JobTool
from time import time
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
    def is_match(self, true_tool_name: str, pred_tool_name: str):
        true_tool_name = true_tool_name.lower().strip()
        pred_tool_name = pred_tool_name.lower().strip()

        return true_tool_name in pred_tool_name or pred_tool_name in true_tool_name
    

    def calc_tools_metrics(self, proposal_tools: list[dict[str, str]], tool_reviews: list[ProposalToolReview]) -> dict[str, float]:
        """
        Args:
            proposal_tools: the given proposal tools representing the ground truth
            tool_reviews  : the agent tool reviews (what we evalaute)
        """
        # calc matches
        matched_true = set()
        matched_pred = set()

        for pred_idx, tool_review in enumerate(tool_reviews):
            for true_idx, proposal_tool in enumerate(proposal_tools):
                pred_tool_name = tool_review.tool_name
                true_tool_name = proposal_tool["tool_name"]

                if true_idx not in matched_true and self.is_match(true_tool_name, pred_tool_name):
                    matched_true.add(true_idx)
                    matched_pred.add(pred_idx)
                    break
        
        # calc scores
        TP = len(matched_pred)                          
        FP = len(tool_reviews) - len(matched_pred)
        FN = len(proposal_tools) - len(matched_true)

        accuracy  = TP / (TP + FP + FN) if (TP + FP + FN) else 0
        precision = TP / (TP + FP)      if (TP + FP)      else 0
        recall    = TP / (TP + FN)      if (TP + FN)      else 0

        return {
            "accuracy"      : accuracy,
            "precision"     : precision,
            "recall"        : recall,
        }
    
    

    def calc_confidence_accuracy(self, proposal_tools: list[dict[str, str]], tool_reviews: list[ProposalToolReview]) -> float:
        matched_pairs = []
        true_indices = set()
        for pred_tool in tool_reviews:
            pred_tool_name = pred_tool.tool_name

            for true_idx, true_tool in enumerate(proposal_tools):
                true_tool_name = true_tool["tool_name"]

                if true_idx not in true_indices and self.is_match(true_tool_name, pred_tool_name):
                    matched_pairs.append((true_tool, pred_tool))
                    true_indices.add(true_idx)
                    break
        
        correct = 0
        total = 0
        for true_tool, pred_tool in matched_pairs:
            true_tool_confidence = true_tool["with_confidence"]
            pred_tool_confidence = pred_tool.with_confidence
            total += 1

            if true_tool_confidence == pred_tool_confidence:
                correct += 1
        
        return (correct / total) if total else 0



    def get_metric_names(self) -> tuple[str, str, str, str]:
        return (
            "tools_analysis_accuracy",
            "tools_analysis_precision",
            "tools_analysis_recall",
            "with_confidence_accuracy",
            "agent_invokation_time"
        )


    def evaluate_sample(self, sample: dict):
        """
        Evaluating the agent on a single sample
        """
        # get data
        job_tools = sample['job_tools']
        job_tools = [JobTool(**job_tool) for job_tool in job_tools]

        # proposals
        proposals = sample["proposals"]

        metrics = {
            metric_name: []
            for metric_name in self.get_metric_names()
        }

        for p in proposals:
            proposal = p["proposal"]
            proposal_tools = p["proposal_tools"]

            formatted_agent_ip = format_ip_for_proposal_tools_analyzer(
                job_tools = job_tools,
                proposal = proposal,
            )
            
            start_time = time() 
            agent_response = self.invoke(input = formatted_agent_ip)
            end_time = time() 

            tool_reviews = agent_response.tool_reviews

            # measure metrics
            tools_metrics = self.calc_tools_metrics(
                proposal_tools = proposal_tools,
                tool_reviews   = tool_reviews 
            )

            confidence_acc = self.calc_confidence_accuracy(
                proposal_tools = proposal_tools,
                tool_reviews   = tool_reviews 
            )

            metrics["tools_analysis_accuracy"].append(tools_metrics["accuracy"])
            metrics["tools_analysis_precision"].append(tools_metrics["precision"])
            metrics["tools_analysis_recall"].append(tools_metrics["recall"])
            metrics["with_confidence_accuracy"].append(confidence_acc)
            metrics["agent_invokation_time"].append(end_time - start_time)


        # average per proposal
        return {
            metric_name: sum(values) / len(values) if values else 0.0
            for metric_name, values in metrics.items()
        }


        
