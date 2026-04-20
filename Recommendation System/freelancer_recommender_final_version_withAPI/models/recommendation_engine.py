"""
RecommendationEngine
====================
High-level facade that orchestrates:
  DataPreprocessor → EmbeddingEngine → VectorStore → ScoringEngine

Usage
-----
    engine = RecommendationEngine()
    engine.build()                               # load data, embed, index

    jobs = engine.recommend_jobs(freelancer_id="abc123", top_n=10)
    freelancers = engine.recommend_freelancers(job_index=0, top_n=10)

    for match in jobs:
        print(match.summary())
"""

import logging

import numpy as np
import pandas as pd

from config.settings import (
    FREELANCERS_CSV, JOBS_CSV,
    DEFAULT_TOP_N, MIN_SCORE_THRESHOLD,
)
from data.preprocessor import DataPreprocessor
from models.embedding_engine import EmbeddingEngine, FreelancerProfile
from models.vector_store import VectorStore
from models.scoring_engine import ScoringEngine, MatchResult

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    End-to-end recommender for freelancer ↔ job matching.

    Parameters
    ----------
    freelancers_path  Path to the freelancers CSV.
    jobs_path         Path to the jobs CSV.
    embedding_model   SentenceTransformer model name.
    """

    def __init__(
        self,
        freelancers_path=FREELANCERS_CSV,
        jobs_path=JOBS_CSV,
        embedding_model: str | None = None,
    ) -> None:
        self._preprocessor = DataPreprocessor(freelancers_path, jobs_path)
        self._embedder      = EmbeddingEngine(
            **({"model_name": embedding_model} if embedding_model else {})
        )
        self._store   = VectorStore()
        self._scorer  = ScoringEngine()

        self.df_freelancers: pd.DataFrame | None = None
        self.df_jobs:        pd.DataFrame | None = None
        self._f_emb: np.ndarray | None = None
        self._j_emb: np.ndarray | None = None
        self._freelancer_id_to_index: dict[str, int] = {}
        self._freelancer_profiles: list[FreelancerProfile] = []

    # ── Setup ─────────────────────────────────────────────────────────────

    def build(self, recreate_index: bool = False) -> "RecommendationEngine":
        """
        Full pipeline: load data → embed → index in Qdrant.
        Safe to call multiple times; incremental updates are handled
        automatically unless recreate_index=True.
        """
        logger.info("=== Building RecommendationEngine ===")

        self.df_freelancers, self.df_jobs = self._preprocessor.load_and_clean()

        cache = self._embedder.load_or_compute(self.df_freelancers, self.df_jobs)
        self._f_emb = cache["freelancer_embeddings"]
        self._j_emb = cache["job_embeddings"]
        self._freelancer_profiles = cache.get("freelancer_profiles", [])

        logger.info("Indexing vectors in Qdrant …")
        self._store.index_jobs(
            self._j_emb, self.df_jobs, recreate=recreate_index
        )
        self._store.index_freelancers(
            self._f_emb, self.df_freelancers, recreate=recreate_index
        )

        # Build fast ID → index lookup
        self._freelancer_id_to_index = {
            fid: idx
            for idx, fid in enumerate(self.df_freelancers["id"].tolist())
        }

        logger.info("=== Engine ready ===")
        return self

    # ── Freelancer Profile (embedding + budget + locations) ───────────────

    def get_freelancer_profile(
        self,
        freelancer_id: str | None = None,
        freelancer_index: int | None = None,
    ) -> FreelancerProfile:
        """
        Return the full FreelancerProfile for a given freelancer.

        The profile contains:
          - embedding              : dense semantic vector (numpy float32)
          - preferred_budget_range : AI-predicted (min_label, max_label) tuple
          - preferred_locations    : AI-predicted list of preferred client countries

        Backend can store preferred_budget_range and preferred_locations in the
        DB and later use them as pre-filters alongside embedding similarity.

        Parameters
        ----------
        freelancer_id    : UUID string from the source data.
        freelancer_index : 0-based row index (alternative to ID).
        """
        idx = self._resolve_freelancer(freelancer_id, freelancer_index)
        if idx >= len(self._freelancer_profiles):
            raise IndexError(f"Profile for index {idx} not found in cache.")
        return self._freelancer_profiles[idx]

    # ── Freelancer → Jobs ─────────────────────────────────────────────────

    def recommend_jobs(
        self,
        freelancer_id: str | None = None,
        freelancer_index: int | None = None,
        top_n: int = DEFAULT_TOP_N,
        min_score: float = MIN_SCORE_THRESHOLD,
    ) -> list[MatchResult]:
        """
        Return the top-N best job matches for a freelancer.
        Provide either freelancer_id (string) or freelancer_index (int).
        """
        idx = self._resolve_freelancer(freelancer_id, freelancer_index)
        freelancer_row = self.df_freelancers.iloc[idx]

        # ANN search — O(log n) via Qdrant HNSW
        raw_results = self._store.search_jobs(
            self._f_emb[idx], top_n=top_n * 2  # over-fetch for re-ranking
        )

        # Hybrid re-ranking
        matches = self._scorer.rank(raw_results, freelancer_row, min_score)
        return matches[:top_n]

    # ── Job → Freelancers (reverse matching) ─────────────────────────────

    def recommend_freelancers(
        self,
        job_index: int,
        top_n: int = DEFAULT_TOP_N,
        min_score: float = MIN_SCORE_THRESHOLD,
    ) -> list[MatchResult]:
        """
        Return the top-N best freelancer matches for a given job.
        (Reverse direction — previously missing from the system.)
        """
        if self._j_emb is None:
            raise RuntimeError("Call build() first.")

        raw_results = self._store.search_freelancers(
            self._j_emb[job_index], top_n=top_n * 2
        )

        job_row = self.df_jobs.iloc[job_index]

        # Reuse ScoringEngine with swapped roles:
        # treat the job's budget / country as the "anchor" for structured score
        results = []
        for point in raw_results:
            pay = point.payload or {}
            sem = float(point.score)
            # Use a minimal Series with the job fields the scorer needs
            proxy = pd.Series({
                "rate_usd":      job_row.get("budget_avg"),
                "feedback_score": pay.get("feedback_score", 0.5),
                "jobs_done":     pay.get("jobs_done", 0),
                "country_code":  job_row.get("country_code", ""),
            })
            match = self._scorer._score(sem, proxy, pay, point.id)
            if match.hybrid_score >= min_score:
                match.job_title = str(pay.get("name", f"Freelancer #{point.id}"))
                results.append(match)

        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results[:top_n]

    # ── Pretty print ──────────────────────────────────────────────────────

    def print_recommendations(
        self,
        matches: list[MatchResult],
        title: str = "Recommendations",
    ) -> None:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
        for i, m in enumerate(matches, 1):
            print(f" {i:>2}. {m.summary()}")
            desc = (m.payload.get("job_description") or m.payload.get("description") or "")
            if desc:
                print(f"      {str(desc)[:120]}…")
        print()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _resolve_freelancer(
        self,
        freelancer_id: str | None,
        freelancer_index: int | None,
    ) -> int:
        if self._f_emb is None:
            raise RuntimeError("Call build() before querying.")
        if freelancer_index is not None:
            return int(freelancer_index)
        if freelancer_id is not None:
            idx = self._freelancer_id_to_index.get(freelancer_id)
            if idx is None:
                raise KeyError(f"Freelancer id '{freelancer_id}' not found.")
            return idx
        raise ValueError("Provide either freelancer_id or freelancer_index.")
