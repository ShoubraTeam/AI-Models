# -----------------------------------------------------------------
# Implementing an Agent to extract tools from the job description
# -----------------------------------------------------------------


from time import time

from models.config.agents_config import PRR_DEFAULT_MODELS_CFG

from agents.proposal_rejection_reasons.base_agent import BaseAgent
from models.schemas import JobToolResponse

class JobToolsExtractor(BaseAgent):
    def __init__(
        self,
        groq_client,
        model_name: str,
        system_prompt: str,
        structured_response = None,
        model_provider: str = "groq",
        **kwargs
    ):
        
        if "temperature" not in kwargs:
            kwargs = PRR_DEFAULT_MODELS_CFG['job_tools_extractor']

        super().__init__(
            groq_client = groq_client,
            model_name = model_name, 
            system_prompt = system_prompt, 
            structured_response = structured_response, 
            model_provider = model_provider,
            **kwargs
        )
    
    # ----------------------------- Modeling -------------------------------- #

    
    def invoke(self, job_desc: str) -> JobToolResponse:
        return super().invoke(input = job_desc)
    
    def ainvoke(self, job_desc: str):
        return super().ainvoke(input = job_desc)
    
    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)
    
    # ---------------------------- Evaluation ---------------------------------- # 
    def is_match(self, true_tool_name: str, pred_tool_name: str):
        true_tool_name = true_tool_name.lower().strip()
        pred_tool_name = pred_tool_name.lower().strip()

        return true_tool_name in pred_tool_name or pred_tool_name in true_tool_name

    def calc_tool_names_metrics(self, true_tool_names: list[str], pred_tool_names: list[str]) -> dict[str, float]:
        """
        Args:
            true_tools: list of true tool names 
            pred_tools: list of pred tool names
        """
        if not true_tool_names and not pred_tool_names:
            return {
                "accuracy" : 1.0,
                "precision": 1.0,
                "recall"   : 1.0,
            }

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
        recall    = TP / (TP + FN)      if (TP + FN)      else 1.0

        return {
            "accuracy"      : accuracy,
            "precision"     : precision,
            "recall"        : recall,
        }
    
    
    def calc_tool_necessity_metrics(self, true_tools: list[dict], pred_tools: list) -> float:
        if not true_tools and not pred_tools:
            return 1.0

        matched_pairs = []
        true_indices = set()
        for pred_tool in pred_tools:
            pred_tool_name = pred_tool.tool_name

            for true_idx, true_tool in enumerate(true_tools):
                true_tool_name = true_tool["tool_name"]

                if true_idx not in true_indices and self.is_match(true_tool_name, pred_tool_name):
                    matched_pairs.append((true_tool, pred_tool))
                    true_indices.add(true_idx)
                    break
        
        correct = 0
        total = 0
        for true_tool, pred_tool in matched_pairs:
            true_tool_level = true_tool["necessity_level"]
            pred_tool_level = pred_tool.necessity_level
            total += 1

            if true_tool_level == pred_tool_level:
                correct += 1
        
        return (correct / total) if total else 0.0
        
    
    def get_metric_names(self) -> tuple[str, str, str, str, str]:
        return (
            "tools_alignment_accuracy",
            "tools_alignment_precision",
            "tools_alignment_recall",
            "tools_necessity_accuracy",
            "agent_invokation_time"
        )

    def evaluate_sample(self, sample: dict):
        """
        Evaluating the agent on a single sample
        """
        # get data
        job_desc = sample["job_desc"]
        true_tools = sample["job_tools"]

        # invoke agent
        start_time = time()
        pred_tools = self.invoke(job_desc = job_desc).tools
        end_time = time() 


        # measure metrics
        true_tool_names = set([tool["tool_name"] for tool in true_tools])
        pred_tool_names = set([tool.tool_name for tool in pred_tools])

        tools_metrics = self.calc_tool_names_metrics(true_tool_names, pred_tool_names)
        necessity_accuracy = self.calc_tool_necessity_metrics(true_tools, pred_tools)

        return {
            "tools_alignment_accuracy"  : tools_metrics["accuracy"],
            "tools_alignment_precision" : tools_metrics["precision"],
            "tools_alignment_recall"    : tools_metrics["recall"],
            "tools_necessity_accuracy"  : necessity_accuracy,
            "agent_invokation_time"     : end_time - start_time
        }
    
    

            


        

        
