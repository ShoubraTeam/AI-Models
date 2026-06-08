from agents.BaseAgent import BaseAgent
from schemas import JobKeyPointsSchema
from prompts import JOB_KEY_POINTS_EXTRACTION_PROMPT
from helpers.config import DEFAULT_MODELS_CFG
from time import time


class JobKeyPointsExtractor(BaseAgent):
    """
    Sub-agent 1: Extracts core_problem, required_deliverables, and key_keywords
    from the job description.

    Designed to be tested and evaluated independently.

    Output: JobKeyPointsSchema
        - core_problem          : str
        - required_deliverables : List[str]
        - key_keywords          : List[str]
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
            kwargs = DEFAULT_MODELS_CFG["job_key_points_extractor"]

        super().__init__(model_name, system_prompt, tools, structured_response, **kwargs)

    def get_agent(self):
        return super().get_agent()

    def invoke(self, input, return_structured_op_only=True):
        return super().invoke(input, return_structured_op_only)

    def validate_agent_output(self, agent_output):
        return super().validate_agent_output(agent_output)

    # ---------------------------- Evaluation ----------------------------

    def get_metric_names(self) -> tuple:
        return (
            "keyword_recall",
            "keyword_precision",
            "deliverable_recall",
            "agent_invocation_time",
        )

    def evaluate_sample(self, sample: dict) -> dict[str, float]:
        """
        Evaluating the JobKeyPointsExtractor on a single sample.

        Sample structure (from EvaluationDataParser.get_job_key_points_extractor_data):
            {
                "job_desc"    : str,
                "core_problem": str,       # ground truth (not scored — hard to compare automatically)
                "deliverables": List[str], # ground truth
                "key_keywords": List[str], # ground truth
            }

        Metrics:
            - keyword_recall    : how many true keywords the agent found
            - keyword_precision : how many of the agent's keywords are actually correct
            - deliverable_recall: how many true deliverables the agent found
            - agent_invocation_time: average invocation time in seconds
        """
        job_desc          = sample.get("job_desc", "")
        true_keywords     = [kw.lower() for kw in sample.get("key_keywords", [])]
        true_deliverables = [d.lower()  for d  in sample.get("deliverables", [])]

        times = []

        start_time = time()
        agent_response: JobKeyPointsSchema = self.invoke(input=job_desc)
        times.append(time() - start_time)

        pred_keywords     = [kw.lower() for kw in agent_response.key_keywords]
        pred_deliverables = [d.lower()  for d  in agent_response.required_deliverables]

        # keyword recall — how many true keywords appeared in predicted keywords
        if true_keywords:
            kw_recalled = sum(
                1 for true_kw in true_keywords
                if any(true_kw in pred_kw or pred_kw in true_kw for pred_kw in pred_keywords)
            )
            keyword_recall = kw_recalled / len(true_keywords)
        else:
            keyword_recall = 0.0

        # keyword precision — how many predicted keywords are actually in true keywords
        if pred_keywords:
            kw_correct = sum(
                1 for pred_kw in pred_keywords
                if any(pred_kw in true_kw or true_kw in pred_kw for true_kw in true_keywords)
            )
            keyword_precision = kw_correct / len(pred_keywords)
        else:
            keyword_precision = 0.0

        # deliverable recall — how many true deliverables appeared in predicted deliverables
        if true_deliverables:
            del_recalled = sum(
                1 for true_d in true_deliverables
                if any(true_d in pred_d or pred_d in true_d for pred_d in pred_deliverables)
            )
            deliverable_recall = del_recalled / len(true_deliverables)
        else:
            deliverable_recall = 0.0

        return {
            "keyword_recall"      : round(keyword_recall,     2),
            "keyword_precision"   : round(keyword_precision,  2),
            "deliverable_recall"  : round(deliverable_recall, 2),
            "agent_invocation_time": round(sum(times) / len(times) if times else 0.0, 2),
        }