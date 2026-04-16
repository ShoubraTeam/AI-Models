"""
ScoringEngine
=============
Post-retrieval re-ranking that blends:

  hybrid_score = w_sem * semantic_score
               + w_struct * (w_rate * rate_compatibility
                           + w_rep  * reputation_score)
               + geo_bonus (if country match)

All weights are configurable in config/settings.py.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config.settings import (
    WEIGHT_SEMANTIC,
    WEIGHT_STRUCTURED,
    WEIGHT_RATE_COMPAT,
    WEIGHT_REPUTATION,
    GEO_BONUS,
    WEEKLY_HOURS,
    MIN_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """A single scored match between a freelancer and a job."""

    job_id: int                        # Qdrant point id (row index)
    job_title: str
    semantic_score: float
    structured_score: float
    hybrid_score: float
    geo_bonus_applied: bool
    payload: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        geo = " (+geo)" if self.geo_bonus_applied else ""
        return (
            f"[{self.hybrid_score:.2%}{geo}] {self.job_title[:60]}"
            f"  (sem={self.semantic_score:.2%}, struct={self.structured_score:.2%})"
        )


class ScoringEngine:
    """
    Re-rank Qdrant search results using hybrid scoring.

    Parameters
    ----------
    w_semantic    Weight for the raw semantic cosine score.
    w_structured  Weight for the structured feature score.
    w_rate        Sub-weight (within structured) for rate compatibility.
    w_reputation  Sub-weight (within structured) for client reputation.
    geo_bonus     Flat bonus added when freelancer and client share a country.
    """

    def __init__(
        self,
        w_semantic:   float = WEIGHT_SEMANTIC,
        w_structured: float = WEIGHT_STRUCTURED,
        w_rate:       float = WEIGHT_RATE_COMPAT,
        w_reputation: float = WEIGHT_REPUTATION,
        geo_bonus:    float = GEO_BONUS,
    ) -> None:
        assert abs(w_semantic + w_structured - 1.0) < 1e-6, \
            "w_semantic + w_structured must equal 1.0"
        assert abs(w_rate + w_reputation - 1.0) < 1e-6, \
            "w_rate + w_reputation must equal 1.0"

        self.w_semantic   = w_semantic
        self.w_structured = w_structured
        self.w_rate       = w_rate
        self.w_reputation = w_reputation
        self.geo_bonus    = geo_bonus

    # ── Public API ────────────────────────────────────────────────────────

    def rank(
        self,
        qdrant_results: list,          # list[ScoredPoint]
        freelancer_row: pd.Series,
        min_score: float = MIN_SCORE_THRESHOLD,
    ) -> list[MatchResult]:
        """
        Apply hybrid scoring to a Qdrant result list and return
        MatchResult objects sorted by hybrid_score descending.
        """
        results = []
        for point in qdrant_results:
            sem   = float(point.score)
            pay   = point.payload or {}
            match = self._score(sem, freelancer_row, pay, point.id)
            if match.hybrid_score >= min_score:
                results.append(match)

        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results

    # ── Scoring sub-components ────────────────────────────────────────────

    def _score(
        self,
        semantic_score: float,
        freelancer: pd.Series,
        job_payload: dict[str, Any],
        job_id: int,
    ) -> MatchResult:
        structured = self._structured_score(freelancer, job_payload)
        geo        = self._geo_match(freelancer, job_payload)

        hybrid = (
            self.w_semantic   * semantic_score
            + self.w_structured * structured
            + (self.geo_bonus if geo else 0.0)
        )
        hybrid = min(hybrid, 1.0)

        return MatchResult(
            job_id=job_id,
            job_title=str(job_payload.get("job_title", "Unknown")),
            semantic_score=semantic_score,
            structured_score=structured,
            hybrid_score=hybrid,
            geo_bonus_applied=geo,
            payload=job_payload,
        )

    def _structured_score(
        self,
        freelancer: pd.Series,
        job: dict[str, Any],
    ) -> float:
        rate_compat = self._rate_compatibility(freelancer, job)
        reputation  = self._reputation_score(freelancer)
        return self.w_rate * rate_compat + self.w_reputation * reputation

    def _rate_compatibility(
        self,
        freelancer: pd.Series,
        job: dict[str, Any],
    ) -> float:
        """
        Compare freelancer hourly rate against implied job rate
        (budget / WEEKLY_HOURS).  Returns 1.0 for a perfect match,
        tapering to 0.0 for a >2× mismatch.
        """
        f_rate = freelancer.get("rate_usd")
        budget = job.get("budget_avg") or job.get("avg_price")

        if not f_rate or not budget:
            return 0.5   # neutral when data is missing

        implied_rate = float(budget) / WEEKLY_HOURS
        if implied_rate <= 0:
            return 0.5

        ratio = f_rate / implied_rate
        # Gaussian decay centred at ratio=1.0, σ≈0.8
        return math.exp(-0.5 * ((math.log(max(ratio, 1e-6))) ** 2) / (0.8 ** 2))

    @staticmethod
    def _reputation_score(freelancer: pd.Series) -> float:
        """
        Blend feedback_score and jobs_done normalised to [0, 1].
        Freelancers with no history get a neutral 0.5.
        """
        feedback  = float(freelancer.get("feedback_score") or 0)
        jobs_done = int(freelancer.get("jobs_done") or 0)

        if feedback == 0 and jobs_done == 0:
            return 0.5

        # Normalise jobs_done: 50+ jobs = full score
        jobs_norm = min(jobs_done / 50, 1.0)
        return 0.7 * feedback + 0.3 * jobs_norm

    @staticmethod
    def _geo_match(freelancer: pd.Series, job: dict[str, Any]) -> bool:
        """True when both freelancer and client share a non-empty country code."""
        f_code = str(freelancer.get("country_code") or "").strip()
        j_code = str(job.get("country_code") or "").strip()
        return bool(f_code and j_code and f_code == j_code)
