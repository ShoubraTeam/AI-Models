from agents.BaseAgent import BaseAgent
from schemas.language_clarity.language_clarity_eval_schema import LanguageClarityEvalSchema
from prompts.language_clarity.language_clarity_evaluator_prompt import LANGUAGE_CLARITY_EVALUATOR_PROMPT
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

    Text metrics (word count, sentence length) are handled in processing.

    Designed to be tested and evaluated independently.

    Output: LanguageClarityEvalSchema
        - is_clear               : bool
        - is_professional        : bool
        - has_misleading_phrasing: bool
        - summary                : str
        - confidence_score       : float
    """

    def __init__(self, model_name: str = CFG.GROQ_LLAMA_70b, temperature: float = None):

        if temperature is None:
            temperature = CFG.MODELS_CFG["language_clarity_pipeline"]["language_clarity_evaluator_temperature"]

        max_tokens = CFG.MODELS_CFG["language_clarity_pipeline"]["language_clarity_evaluator_max_tokens"]

        super().__init__(
            model_name=model_name,
            system_prompt=LANGUAGE_CLARITY_EVALUATOR_PROMPT,
            model_provider=CFG.PROVIDER_GROQ,
            structured_response=LanguageClarityEvalSchema,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def invoke(self, proposal_text: str) -> LanguageClarityEvalSchema:
        """
        Args:
            proposal_text: The freelancer's proposal text only.
                           No job description needed for this agent.
        """
        return super().invoke(input=proposal_text)
