from agents.BaseAgent import BaseAgent
from schemas.language_clarity.language_clarity_eval_schema import LanguageClarityEvalSchema
from prompts import LANGUAGE_CLARITY_EVALUATOR_PROMPT
import helpers.config as CFG


class LanguageClarityEvaluator(BaseAgent):
    """
    Sub-agent responsible for evaluating the language quality of a proposal.

    Unlike other agents, this one does NOT need the job description —
    it evaluates the proposal text alone.

    Answers exactly 3 questions:
        - is_clear               : is the proposal easy to understand?
        - is_professional        : is the tone appropriate for a client?
        - has_misleading_phrasing: are there vague or empty promises?

    Text metrics (word count, sentence length, grammar) are handled in processing.

    Designed to be tested and evaluated independently.

    Output: LanguageClarityEvalSchema
        - is_clear               : bool
        - is_professional        : bool
        - has_misleading_phrasing: bool
        - summary                : str
        - confidence_score       : float
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        tools: list = [],
        structured_response = None,
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = CFG.DEFAULT_MODELS_CFG["language_clarity_evaluator"]

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)

    def invoke(self, proposal_text: str) -> LanguageClarityEvalSchema:
        """
        Args:
            proposal_text: The freelancer's proposal text only.
                           No job description needed for this agent.
        """
        return super().invoke(input=proposal_text)
        # ---------------------------- Evaluation ----------------------------
    
    def calc_binary_metrics(self, true_val: bool, pred_val: bool) -> dict:
        """حساب True Positive, False Positive... إلخ لـ Boolean flag"""
        if true_val is None or pred_val is None:
            return {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
            
        return {
            "TP": 1 if true_val and pred_val else 0,
            "FP": 1 if not true_val and pred_val else 0,
            "TN": 1 if not true_val and not pred_val else 0,
            "FN": 1 if true_val and not pred_val else 0,
        }

    def get_metric_names(self) -> tuple[str, str, str, str, str, str]:
        return (
            "clarity_accuracy",
            "professionalism_accuracy",
            "misleading_phrasing_accuracy",
            "overall_language_accuracy",
            "confidence_score_error",  
            "agent_invocation_time"
        )

    def evaluate_sample(self, sample: dict) -> dict[str, float]:
        """
        Evaluating the LanguageClarityEvaluator on a single sample (contains multiple proposals)
        """
        proposals = sample["proposals"]
        
        # تجميع العدادات لحساب الـ Accuracy الإجمالي في النهاية
        stats = {
            "clarity": {"correct": 0, "total": 0},
            "professional": {"correct": 0, "total": 0},
            "misleading": {"correct": 0, "total": 0}
        }
        
        times = []
        confidence_errors = []

        for p in proposals:
            proposal_text = p["proposal"]
            
            # Ground Truth Data
            true_clear = p.get("is_clear")
            true_prof = p.get("is_professional")
            true_misleading = p.get("has_misleading") 
            
            # Invoke Agent & Measure Time
            start_time = time()
            agent_response = self.invoke(proposal_text=proposal_text)
            end_time = time()
            
            times.append(end_time - start_time)
            
            # Extract Predictions
            pred_clear = agent_response.is_clear
            pred_prof = agent_response.is_professional
            pred_misleading = agent_response.has_misleading_phrasing
            
            # Accumulate correctness
            if true_clear is not None:
                stats["clarity"]["total"] += 1
                if true_clear == pred_clear: stats["clarity"]["correct"] += 1
                
            if true_prof is not None:
                stats["professional"]["total"] += 1
                if true_prof == pred_prof: stats["professional"]["correct"] += 1
                
            if true_misleading is not None:
                stats["misleading"]["total"] += 1
                if true_misleading == pred_misleading: stats["misleading"]["correct"] += 1
            
            if hasattr(agent_response, 'confidence_score') and agent_response.confidence_score is not None:
                pass

        # حساب الـ Accuracies
        clarity_acc = stats["clarity"]["correct"] / stats["clarity"]["total"] if stats["clarity"]["total"] else 0.0
        prof_acc = stats["professional"]["correct"] / stats["professional"]["total"] if stats["professional"]["total"] else 0.0
        misleading_acc = stats["misleading"]["correct"] / stats["misleading"]["total"] if stats["misleading"]["total"] else 0.0
        
        # المتوسط الإجمالي للـ 3 flags
        total_correct = stats["clarity"]["correct"] + stats["professional"]["correct"] + stats["misleading"]["correct"]
        total_flags = stats["clarity"]["total"] + stats["professional"]["total"] + stats["misleading"]["total"]
        overall_acc = total_correct / total_flags if total_flags else 0.0

        return {
            "clarity_accuracy": clarity_acc,
            "professionalism_accuracy": prof_acc,
            "misleading_phrasing_accuracy": misleading_acc,
            "overall_language_accuracy": overall_acc,
            "agent_invocation_time": sum(times) / len(times) if times else 0.0
        }
