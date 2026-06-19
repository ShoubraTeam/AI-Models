
from time import time

from models.config.agents_config import PRR_DEFAULT_MODELS_CFG

from models.schemas import JobUnderstandingEvalSchema
from agents.proposal_rejection_reasons.base_agent import BaseAgent

class JobUnderstandingEvaluator(BaseAgent):
    """
    Evaluates the proposal against the extracted job key points.

    Very task-specific — answers exactly 3 boolean questions:
        - problem_identified
        - solution_proposed
        - practical_steps_mentioned

    Everything else (keyword matching, scoring, similarity) is handled
    by the processing layer using normal code metrics.

    Designed to be tested and evaluated independently.

    Output: JobUnderstandingEvalSchema
        - problem_identified        : bool
        - solution_proposed         : bool
        - practical_steps_mentioned : bool
        - matched_keywords          : list[str]
        - missing_keywords          : list[str]
        - summary                   : str
        - confidence_score          : float
    """

    def __init__(
        self,
        groq_client,
        model_name: str,
        system_prompt: str,
        structured_response: JobUnderstandingEvalSchema = None,
        model_provider: str = "groq",
        **kwargs
    ):
        if "temperature" not in kwargs:
            kwargs = PRR_DEFAULT_MODELS_CFG["job_understanding_evaluator"]

        super().__init__(
            groq_client = groq_client,
            model_name = model_name, 
            system_prompt = system_prompt, 
            structured_response = structured_response, 
            model_provider = model_provider,
            **kwargs
        )

    # -------------------------- Modeling ---------------------- #
    
    def prepare_job_undertanding_evaluator_ip(
        self,
        core_problem         : str,
        required_deliverables: list[str],
        key_keywords         : list[str],
        proposal_text        : str
    ) -> str:

        required_deliverables_str = ""
        for deliverable in required_deliverables:
            required_deliverables_str += f"\t- {deliverable}"

        key_keywords_str = ", ".join(key_keywords)

        return (
            f"Core Problem:\n{core_problem}\n\n"
            f"Required Deliverables:\n{required_deliverables_str}\n\n"
            f"Key Keywords:\n{key_keywords_str}\n\n"
            f"Freelancer Proposal:\n{proposal_text}"
        )

    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)

    def invoke(
        self,
        core_problem: str,
        required_deliverables: list[str],
        key_keywords: list[str],
        proposal_text: str
    ) -> JobUnderstandingEvalSchema:
        """
        Keyword matching is done by the LLM, so the original keyword list must
        be included in the input and echoed through matched/missing fields.

        Args:
            core_problem          : Extracted core problem from JobKeyPointsExtractor.
            required_deliverables : Extracted deliverables from JobKeyPointsExtractor.
            key_keywords          : Extracted key terms to classify as matched or missing.
            proposal_text         : The freelancer's proposal text.
        """
        formatted_input = self.prepare_job_undertanding_evaluator_ip(
            core_problem          = core_problem,
            required_deliverables = required_deliverables,
            key_keywords          = key_keywords,
            proposal_text         = proposal_text
        ) 
        
        return super().invoke(input = formatted_input)


    async def ainvoke(
        self,
        core_problem         : str,
        required_deliverables: list[str],
        key_keywords         : list[str],
        proposal_text        : str
    ) -> JobUnderstandingEvalSchema:
        formatted_input = self.prepare_job_undertanding_evaluator_ip(
            core_problem          = core_problem,
            required_deliverables = required_deliverables,
            key_keywords          = key_keywords,
            proposal_text         = proposal_text
        ) 

        return await super().ainvoke(input = formatted_input)

    # ---------------------------- Evaluation ----------------------------

    def get_metric_names(self) -> tuple:
        return (
            "problem_identified_accuracy",
            "solution_proposed_accuracy",
            "practical_steps_accuracy",
            "overall_understanding_accuracy",
            "agent_invocation_time",
        )

    def evaluate_sample(self, sample: dict) -> dict[str, float]:
        """
        Evaluating the JobUnderstandingEvaluator on a single sample.

        Sample structure (from EvaluationDataParser.get_job_understanding_evaluator_data):
            {
                "job_desc": str,
                "job_data": {
                    "core_problem"         : str,
                    "required_deliverables": list[str],
                    "key_keywords"         : list[str],
                },
                "proposals": [
                    {
                        "proposal"               : str,
                        "true_problem_identified": bool,
                        "true_solution_proposed" : bool,
                        "true_practical_steps"   : bool,
                        "true_matched_keywords"  : list[str],
                        "true_missing_keywords"  : list[str],
                    },
                    ...
                ]
            }
        """
        core_problem          = sample["job_data"]["core_problem"]
        required_deliverables = sample["job_data"]["required_deliverables"]
        proposals             = sample["proposals"]

        stats = {
            "problem" : {"correct": 0, "total": 0},
            "solution": {"correct": 0, "total": 0},
            "steps"   : {"correct": 0, "total": 0},
        }
        times = []

        for p in proposals:
            proposal_text  = p["proposal"]

            # ground truth
            true_problem   = p.get("true_problem_identified")
            true_solution  = p.get("true_solution_proposed")
            true_steps     = p.get("true_practical_steps")

            # invoke
            start_time     = time()
            agent_response = self.invoke(
                core_problem          = core_problem,
                required_deliverables = required_deliverables,
                key_keywords          = sample["job_data"].get("key_keywords", []),
                proposal_text         = proposal_text,
            )
            times.append(time() - start_time)

            # predictions
            pred_problem   = agent_response.problem_identified
            pred_solution  = agent_response.solution_proposed
            pred_steps     = agent_response.practical_steps_mentioned

            # accumulate
            if true_problem is not None:
                stats["problem"]["total"] += 1
                if true_problem == pred_problem:
                    stats["problem"]["correct"] += 1

            if true_solution is not None:
                stats["solution"]["total"] += 1
                if true_solution == pred_solution:
                    stats["solution"]["correct"] += 1

            if true_steps is not None:
                stats["steps"]["total"] += 1
                if true_steps == pred_steps:
                    stats["steps"]["correct"] += 1

        # per-flag accuracy
        prob_acc  = stats["problem"]["correct"]  / stats["problem"]["total"]  if stats["problem"]["total"]  else 0.0
        sol_acc   = stats["solution"]["correct"] / stats["solution"]["total"] if stats["solution"]["total"] else 0.0
        steps_acc = stats["steps"]["correct"]    / stats["steps"]["total"]    if stats["steps"]["total"]    else 0.0

        # overall accuracy across all 3 flags
        total_correct = stats["problem"]["correct"]  + stats["solution"]["correct"] + stats["steps"]["correct"]
        total_flags   = stats["problem"]["total"]    + stats["solution"]["total"]   + stats["steps"]["total"]
        overall_acc   = total_correct / total_flags if total_flags else 0.0

        return {
            "problem_identified_accuracy"   : round(prob_acc,   2),
            "solution_proposed_accuracy"    : round(sol_acc,    2),
            "practical_steps_accuracy"      : round(steps_acc,  2),
            "overall_understanding_accuracy": round(overall_acc, 2),
            "agent_invocation_time"         : round(sum(times) / len(times) if times else 0.0, 2),
        }
