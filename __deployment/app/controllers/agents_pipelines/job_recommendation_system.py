# -----------------------------------------------
# Required workflow for job_recommendation_system
# -----------------------------------------------

import re
import unicodedata

from agents.recommendation_system import RSEmbeddingEngine
from models.data_config import RS_FREELANCER_EMBEDDING, RS_JOB_EMBEDDING


class JobRecommendationSystemPipeline:
    """
    Job Recommendation System Pipeline.

    Required Methods:
        preprocess(input)        : pre-process the input before calling the agent. If not pre-processing required -> return the input
        call(input)              : invoke/call the agent on the given input
        postprocess(agent_output): post-process the agent output. If no post-processing required -> return the agent_output.

    Args:
        agents (dict): must contain key "RS_embedder" -> RSEmbeddingEngine instance.
        task   (str) : RS_freelancer_embedding | RS_job_embedding
    """

    def __init__(self, agents: dict, task: str) -> None:
        self.embedder: RSEmbeddingEngine = agents["RS_embedder"]
        self.task = task

    # ------------------------------------------------------------------
    # Driver methods
    # ------------------------------------------------------------------

    def preprocess(self, input: tuple[str, list[str], str]) -> str:
        if self.task == RS_FREELANCER_EMBEDDING:
            return self._freelancer_preprocess(input)
        elif self.task == RS_JOB_EMBEDDING:
            return self._job_preprocess(input)
        return input

    def call(self, input: str) -> tuple[str, list[float]]:
        if self.task == RS_FREELANCER_EMBEDDING:
            return self._freelancer_call(input)
        elif self.task == RS_JOB_EMBEDDING:
            return self._job_call(input)
        return input

    def postprocess(self, agent_output: tuple[str, list[float]]) -> dict:
        if self.task == RS_FREELANCER_EMBEDDING:
            return self._freelancer_postprocess(agent_output)
        elif self.task == RS_JOB_EMBEDDING:
            return self._job_postprocess(agent_output)
        return agent_output

    # ------------------------------------------------------------------
    # Freelancer steps
    # ------------------------------------------------------------------

    def _freelancer_preprocess(self, input: tuple[str, list[str], str]) -> str:
        bio, skills, job_title = input
        skills_str = ", ".join(s.strip() for s in skills if s.strip())
        raw = f"{job_title} | {skills_str} | {bio}"
        return self._clean(raw)

    def _freelancer_call(self, input: str) -> tuple[str, list[float]]:
        return (input, self.embedder.embed(input))

    def _freelancer_postprocess(self, agent_output: tuple[str, list[float]]) -> dict:
        enriched_text, embedding = agent_output
        return {"enriched_text": enriched_text, "embedding": embedding}

    # ------------------------------------------------------------------
    # Job steps
    # ------------------------------------------------------------------

    def _job_preprocess(self, input: tuple[str, list[str], str]) -> str:
        description, skills, job_title = input
        skills_str = ", ".join(s.strip() for s in skills if s.strip())
        raw = f"{job_title} | {skills_str} | {description}"
        return self._clean(raw)

    def _job_call(self, input: str) -> tuple[str, list[float]]:
        return (input, self.embedder.embed(input))

    def _job_postprocess(self, agent_output: tuple[str, list[float]]) -> dict:
        enriched_text, embedding = agent_output
        return {"enriched_text": enriched_text, "embedding": embedding}

    # ------------------------------------------------------------------
    # Shared text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", errors="ignore").decode("ascii")
        text = re.sub(r"[^\w\s,.\-|#@%$&]", " ", text)
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"(#\w+\s*)+", " ", text)
        return text.strip().lower()