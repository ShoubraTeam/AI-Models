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