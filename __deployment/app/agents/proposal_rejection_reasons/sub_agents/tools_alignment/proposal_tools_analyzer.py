# -----------------------------------------------------------------
# Implementing an Agent to extract tools from the job description
# -----------------------------------------------------------------

from time import time

from models.config.agents_config import PRR_DEFAULT_MODELS_CFG

from agents.proposal_rejection_reasons.base_agent import BaseAgent
from models.schemas import ProposalToolReview, JobTool, ProposalToolsResponse

class ProposalToolsAnalyzer(BaseAgent):
    def __init__(
        self,
        groq_client,
        model_name: str,
        system_prompt: str,
        structured_response = None,
        **kwargs
    ):
        
        if "temperature" not in kwargs:
            kwargs = PRR_DEFAULT_MODELS_CFG["proposal_tools_analyzer"]

        super().__init__(
            groq_client = groq_client,
            model_name = model_name, 
            system_prompt = system_prompt, 
            structured_response = structured_response, 
            **kwargs
        )
    
    def prepare_proposal_tools_analyzer_ip(
        self,
        job_tools: list[JobTool],
        proposal: str
    ) -> str:
        """
        Preparing data for the agent that analyzes proposal tools

        Args:
            job_tools (list): the list of the tools extracted from the job_description
            proposal  (str) : the proposal text

        return:
            formatted (str): the prepared input formatted as pure string
        """

        formatted = "Job_Tools_List:\n"

        # add tools
        for idx, tool in enumerate(job_tools, start = 1):
            tool_text = f"Tool {idx} => name: {tool.tool_name}, necessity_level: {tool.necessity_level}\n"
            formatted += tool_text

        # add proposal
        formatted += f"\nProposal:\n{proposal}"
        
        return formatted

    
    def invoke(self, job_tools: list[JobTool], proposal: str) -> ProposalToolsResponse:
        fromatted = self.prepare_proposal_tools_analyzer_ip(job_tools, proposal)
        return super().invoke(input = fromatted)
    
    def ainvoke(self, job_tools: list[JobTool], proposal: str) -> ProposalToolsResponse:
        fromatted = self.prepare_proposal_tools_analyzer_ip(job_tools, proposal)
        return super().ainvoke(input = fromatted)

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
        predicted_reviews = [
            tool_review
            for tool_review in tool_reviews
            if tool_review.found_in_proposal is True
        ]

        if not proposal_tools and not predicted_reviews:
            return {
                "accuracy" : 1.0,
                "precision": 1.0,
                "recall"   : 1.0,
            }

        # calc matches
        matched_true = set()
        matched_pred = set()

        for pred_idx, tool_review in enumerate(predicted_reviews):
            for true_idx, proposal_tool in enumerate(proposal_tools):
                pred_tool_name = tool_review.tool_name
                true_tool_name = proposal_tool["tool_name"]

                if true_idx not in matched_true and self.is_match(true_tool_name, pred_tool_name):
                    matched_true.add(true_idx)
                    matched_pred.add(pred_idx)
                    break
        
        # calc scores
        TP = len(matched_pred)                          
        FP = len(predicted_reviews) - len(matched_pred)
        FN = len(proposal_tools) - len(matched_true)

        accuracy  = TP / (TP + FP + FN) if (TP + FP + FN) else 0
        precision = TP / (TP + FP)      if (TP + FP)      else 0
        recall    = TP / (TP + FN)      if (TP + FN)      else 1.0

        return {
            "accuracy"      : accuracy,
            "precision"     : precision,
            "recall"        : recall,
        }
    
    

    def calc_confidence_accuracy(self, proposal_tools: list[dict[str, str]], tool_reviews: list[ProposalToolReview]) -> float:
        predicted_reviews = [
            tool_review
            for tool_review in tool_reviews
            if tool_review.found_in_proposal is True
        ]

        if not proposal_tools and not predicted_reviews:
            return 1.0

        matched_pairs = []
        true_indices = set()
        for pred_tool in predicted_reviews:
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
            proposal_tools = [
                proposal_tool
                for proposal_tool in proposal_tools
                if any(
                    self.is_match(proposal_tool["tool_name"], job_tool.tool_name)
                    for job_tool in job_tools
                )
            ]

            
            start_time = time() 
            agent_response = self.invoke(job_tools = job_tools, proposal = proposal)
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


        
