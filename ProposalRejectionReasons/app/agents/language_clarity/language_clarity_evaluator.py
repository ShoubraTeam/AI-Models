from agents.BaseAgent import BaseAgent
from schemas.language_clarity.language_clarity_eval_schema import LanguageClarityEvalSchema
from prompts import LANGUAGE_CLARITY_EVALUATOR_PROMPT
import helpers.config as CFG
from time import time


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
        structured_response=None,
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

    def get_metric_names(self) -> tuple:
        return (
            "clarity_accuracy",
            "professionalism_accuracy",
            "misleading_phrasing_accuracy",
            "overall_language_accuracy",
            "agent_invocation_time",
        )

    def evaluate_sample(self, sample: dict) -> dict[str, float]:
        """
        Evaluating the LanguageClarityEvaluator on a single sample (contains multiple proposals).

        Sample structure (from EvaluationDataParser.get_language_clarity_evaluator_data):
            {
                "proposals": [
                    {
                        "proposal"       : str,
                        "is_clear"       : bool,
                        "is_professional": bool,
                        "has_misleading" : bool,
                    },
                    ...
                ]
            }
        """
        proposals = sample["proposals"]

        stats = {
            "clarity"    : {"correct": 0, "total": 0},
            "professional": {"correct": 0, "total": 0},
            "misleading" : {"correct": 0, "total": 0},
        }
        times = []

        for p in proposals:
            proposal_text = p["proposal"]

            # ground truth
            true_clear      = p.get("is_clear")
            true_prof       = p.get("is_professional")
            true_misleading = p.get("has_misleading")

            # invoke
            start_time      = time()
            agent_response  = self.invoke(proposal_text=proposal_text)
            times.append(time() - start_time)

            # predictions
            pred_clear      = agent_response.is_clear
            pred_prof       = agent_response.is_professional
            pred_misleading = agent_response.has_misleading_phrasing

            # accumulate
            if true_clear is not None:
                stats["clarity"]["total"] += 1
                if true_clear == pred_clear:
                    stats["clarity"]["correct"] += 1

            if true_prof is not None:
                stats["professional"]["total"] += 1
                if true_prof == pred_prof:
                    stats["professional"]["correct"] += 1

            if true_misleading is not None:
                stats["misleading"]["total"] += 1
                if true_misleading == pred_misleading:
                    stats["misleading"]["correct"] += 1

        # per-flag accuracy
        clarity_acc     = stats["clarity"]["correct"]      / stats["clarity"]["total"]      if stats["clarity"]["total"]      else 0.0
        prof_acc        = stats["professional"]["correct"]  / stats["professional"]["total"]  if stats["professional"]["total"]  else 0.0
        misleading_acc  = stats["misleading"]["correct"]    / stats["misleading"]["total"]    if stats["misleading"]["total"]    else 0.0

        # overall accuracy across all 3 flags
        total_correct   = stats["clarity"]["correct"] + stats["professional"]["correct"] + stats["misleading"]["correct"]
        total_flags     = stats["clarity"]["total"]   + stats["professional"]["total"]   + stats["misleading"]["total"]
        overall_acc     = total_correct / total_flags if total_flags else 0.0

        return {
            "clarity_accuracy"            : round(clarity_acc,    2),
            "professionalism_accuracy"    : round(prof_acc,        2),
            "misleading_phrasing_accuracy": round(misleading_acc,  2),
            "overall_language_accuracy"   : round(overall_acc,     2),
            "agent_invocation_time"       : round(sum(times) / len(times) if times else 0.0, 2),
        }
