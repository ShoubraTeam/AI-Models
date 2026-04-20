"""
EmbeddingEngine
===============
Wraps SentenceTransformer encoding with:
  - Incremental update  (only re-encode new IDs)
  - Pickle cache        (load_or_compute avoids redundant GPU work)
  - Model benchmarking  (compare multiple models on a sample)
  - FreelancerProfile   (embedding + AI-predicted budget range + preferred locations)
"""

import pickle
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from config.settings import (
    EMBEDDING_MODEL, BATCH_SIZE, EMBEDDING_DIM, EMBEDDINGS_CACHE,
)

logger = logging.getLogger(__name__)

# ── Budget bucket boundaries (USD) ────────────────────────────────────────────
# Derived from the jobs dataset percentile distribution:
#   p25 ≈ $20  |  p50 ≈ $140  |  p75 ≈ $1,050
_BUDGET_BINS   = [0, 50, 250, 1_000, 5_000, float("inf")]
_BUDGET_LABELS = ["micro", "small", "medium", "large", "enterprise"]


@dataclass
class FreelancerProfile:
    """
    Enriched output returned by :meth:`EmbeddingEngine.embed_freelancer`.

    Attributes
    ----------
    freelancer_id       : Unique ID from the source data.
    embedding           : Dense float32 vector (shape: [EMBEDDING_DIM]).
    preferred_budget_range : AI-predicted budget bucket, e.g. ``("small", "medium")``.
                          Derived from the freelancer's hour_rate / earnings history
                          and the overlapping job-budget distribution.
    preferred_locations : AI-predicted list of country names the freelancer is
                          likely to accept work from (own country first, then the
                          top client countries that post jobs in their budget tier).
    """
    freelancer_id: str
    embedding: np.ndarray
    preferred_budget_range: Tuple[str, str]   # (min_label, max_label)
    preferred_locations: List[str]


def _infer_budget_range(rate_usd: Optional[float]) -> Tuple[str, str]:
    """
    Map a freelancer's hourly rate to a (min, max) budget-bucket pair.

    Logic
    -----
    A freelancer charging $X/hr typically targets fixed-price projects whose
    total budget covers at least ~2 hrs of their time (lower bound) and up
    to ~20 hrs (upper bound).  We then snap those dollar amounts to the
    nearest predefined bucket.

    Falls back to ("micro", "large") when rate is unknown.
    """
    if rate_usd is None or rate_usd <= 0:
        return ("micro", "large")

    low_usd  = rate_usd * 2          # minimum viable project
    high_usd = rate_usd * 20         # comfortable ceiling

    def _bucket(val: float) -> str:
        for i, (lo, hi) in enumerate(zip(_BUDGET_BINS, _BUDGET_BINS[1:])):
            if lo <= val < hi:
                return _BUDGET_LABELS[i]
        return _BUDGET_LABELS[-1]

    return (_bucket(low_usd), _bucket(high_usd))


def _infer_preferred_locations(
    freelancer_location: str,
    budget_range: Tuple[str, str],
    df_jobs: Optional[pd.DataFrame] = None,
    top_n: int = 5,
) -> List[str]:
    """
    Predict which client countries a freelancer prefers to work with.

    Strategy
    --------
    1. Always include the freelancer's own country (home-market preference).
    2. If the jobs DataFrame is available, find the top-N client countries
       that post jobs within the freelancer's predicted budget tier — these
       represent the most realistic demand pool.
    3. Fall back to a sensible global default list when no data is available.
    """
    home = str(freelancer_location).strip() if freelancer_location else ""
    locations: List[str] = []
    if home and home.lower() not in ("nan", "none", ""):
        locations.append(home)

    if df_jobs is not None and "budget_avg" in df_jobs.columns and "client_country" in df_jobs.columns:
        # Map budget labels back to dollar thresholds for filtering
        label_to_lo = {label: lo for label, lo in zip(_BUDGET_LABELS, _BUDGET_BINS)}
        label_to_hi = {label: hi for label, hi in zip(_BUDGET_LABELS, _BUDGET_BINS[1:])}

        min_label, max_label = budget_range
        lo_thresh = label_to_lo.get(min_label, 0)
        hi_thresh = label_to_hi.get(max_label, float("inf"))

        mask = (df_jobs["budget_avg"] >= lo_thresh) & (df_jobs["budget_avg"] < hi_thresh)
        tier_jobs = df_jobs[mask]

        top_countries = (
            tier_jobs["client_country"]
            .dropna()
            .value_counts()
            .head(top_n + 1)   # +1 to account for possible home dedup
            .index.tolist()
        )
        for country in top_countries:
            c = str(country).strip()
            if c and c not in locations:
                locations.append(c)
                if len(locations) >= top_n:
                    break
    else:
        # Sensible global fallback
        fallback = ["United States", "United Kingdom", "Australia", "Canada", "Germany"]
        for c in fallback:
            if c not in locations:
                locations.append(c)

    return locations[:top_n]


class EmbeddingEngine:
    """Encode text with a SentenceTransformer; cache results on disk."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        cache_path: str | Path = EMBEDDINGS_CACHE,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.model_name  = model_name
        self.cache_path  = Path(cache_path)
        self.batch_size  = batch_size

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model %s on %s", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._device = device

    # ── Public API ─────────────────────────────────────────────────────────

    def embed_freelancer(
        self,
        freelancer_row: pd.Series,
        df_jobs: Optional[pd.DataFrame] = None,
    ) -> FreelancerProfile:
        """
        Embed a single freelancer and return a :class:`FreelancerProfile`
        that contains:

        - **embedding**              – dense semantic vector
        - **preferred_budget_range** – AI-predicted (min, max) budget label tuple
        - **preferred_locations**    – AI-predicted list of preferred client countries

        Parameters
        ----------
        freelancer_row : A single row from the freelancers DataFrame (after
                         preprocessing), or any Series with at least the fields
                         ``id``, ``enriched_text``, ``rate_usd``, and ``location``.
        df_jobs        : Optional jobs DataFrame used to anchor the location
                         prediction to real demand.  When omitted a global
                         fallback list is used.

        Returns
        -------
        FreelancerProfile
        """
        text = str(freelancer_row.get("enriched_text", "") or "")
        emb  = self.encode([text], desc="Single freelancer embed")[0]

        rate        = freelancer_row.get("rate_usd")
        rate_float  = float(rate) if rate is not None and not pd.isna(rate) else None
        budget_rng  = _infer_budget_range(rate_float)

        location    = str(freelancer_row.get("location", "") or "")
        locations   = _infer_preferred_locations(location, budget_rng, df_jobs)

        return FreelancerProfile(
            freelancer_id       = str(freelancer_row.get("id", "")),
            embedding           = emb.astype(np.float32),
            preferred_budget_range = budget_rng,
            preferred_locations = locations,
        )

    def load_or_compute(
        self,
        df_freelancers: pd.DataFrame,
        df_jobs: pd.DataFrame,
    ) -> dict:
        """
        Return embeddings dict, loading from cache when possible and
        only re-encoding records whose IDs are not yet cached.
        """
        cache = self._load_cache()

        if cache is None:
            logger.info("No cache found — computing all embeddings from scratch.")
            cache = self._compute_all(df_freelancers, df_jobs)
        else:
            cache = self._incremental_update(cache, df_freelancers, df_jobs)

        self._save_cache(cache)
        return cache

    def encode(self, texts: list[str], desc: str = "Encoding") -> np.ndarray:
        """Encode a list of strings, returning a float32 ndarray."""
        logger.info("%s: %d texts with batch_size=%d", desc, len(texts), self.batch_size)
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def benchmark_models(
        self,
        df_freelancers: pd.DataFrame,
        df_jobs: pd.DataFrame,
        model_names: list[str],
        sample_f: int = 100,
        sample_j: int = 300,
    ) -> pd.DataFrame:
        """
        Encode samples with each model and return a DataFrame of
        mean pairwise similarity (a proxy for embedding quality).
        Use EvaluationEngine.evaluate_ndcg() for a proper metric.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        rows = []
        f_texts = df_freelancers["enriched_text"].astype(str).tolist()[:sample_f]
        j_texts = df_jobs["enriched_text"].astype(str).tolist()[:sample_j]

        for name in model_names:
            logger.info("Benchmarking model: %s", name)
            model = SentenceTransformer(name, device=self._device)
            f_emb = model.encode(f_texts, batch_size=self.batch_size,
                                 show_progress_bar=False)
            j_emb = model.encode(j_texts, batch_size=self.batch_size,
                                 show_progress_bar=False)
            sims = cosine_similarity(f_emb, j_emb)
            top5 = np.sort(sims, axis=1)[:, -5:]
            rows.append({
                "model":         name,
                "mean_top5_sim": float(np.mean(top5)),
                "dim":           f_emb.shape[1],
            })

        return pd.DataFrame(rows).sort_values("mean_top5_sim", ascending=False)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _compute_all(
        self,
        df_freelancers: pd.DataFrame,
        df_jobs: pd.DataFrame,
    ) -> dict:
        f_emb = self.encode(
            df_freelancers["enriched_text"].astype(str).tolist(),
            desc="Freelancer embeddings",
        )
        j_emb = self.encode(
            df_jobs["enriched_text"].astype(str).tolist(),
            desc="Job embeddings",
        )

        logger.info("Computing freelancer profiles (budget + location inference) ...")
        profiles = self._build_profiles(df_freelancers, f_emb, df_jobs)

        return {
            "freelancer_embeddings":  f_emb,
            "job_embeddings":         j_emb,
            "freelancer_ids":         df_freelancers["id"].tolist(),
            "job_ids":                df_jobs["projectId"].tolist(),
            "model_name":             self.model_name,
            # NEW: one FreelancerProfile per freelancer row
            "freelancer_profiles":    profiles,
        }

    def _incremental_update(
        self,
        cache: dict,
        df_freelancers: pd.DataFrame,
        df_jobs: pd.DataFrame,
    ) -> dict:
        """Re-encode only rows whose IDs are absent from the cache."""
        cached_f_ids = set(cache.get("freelancer_ids", []))
        cached_j_ids = set(cache.get("job_ids", []))

        new_f = df_freelancers[~df_freelancers["id"].isin(cached_f_ids)]
        new_j = df_jobs[~df_jobs["projectId"].isin(cached_j_ids)]

        if len(new_f):
            logger.info("Encoding %d new freelancers", len(new_f))
            new_emb = self.encode(
                new_f["enriched_text"].astype(str).tolist(),
                desc="New freelancer embeddings",
            )
            cache["freelancer_embeddings"] = np.vstack(
                [cache["freelancer_embeddings"], new_emb]
            )
            cache["freelancer_ids"] = cache["freelancer_ids"] + new_f["id"].tolist()

            # Append profiles for newly added freelancers
            new_profiles = self._build_profiles(new_f.reset_index(drop=True), new_emb, df_jobs)
            existing_profiles = cache.get("freelancer_profiles", [])
            cache["freelancer_profiles"] = existing_profiles + new_profiles

        if len(new_j):
            logger.info("Encoding %d new jobs", len(new_j))
            new_emb = self.encode(
                new_j["enriched_text"].astype(str).tolist(),
                desc="New job embeddings",
            )
            cache["job_embeddings"] = np.vstack(
                [cache["job_embeddings"], new_emb]
            )
            cache["job_ids"] = cache["job_ids"] + new_j["projectId"].tolist()

        if not new_f.empty or not new_j.empty:
            logger.info("Cache updated: +%d freelancers, +%d jobs", len(new_f), len(new_j))
        else:
            logger.info("Cache is up-to-date. No re-encoding needed.")

        return cache

    def _build_profiles(
        self,
        df_freelancers: pd.DataFrame,
        embeddings: np.ndarray,
        df_jobs: pd.DataFrame,
    ) -> List[FreelancerProfile]:
        """Build a FreelancerProfile for every row in df_freelancers."""
        profiles: List[FreelancerProfile] = []
        for i, (_, row) in enumerate(df_freelancers.iterrows()):
            rate      = row.get("rate_usd")
            rate_f    = float(rate) if rate is not None and not pd.isna(rate) else None
            budget    = _infer_budget_range(rate_f)
            location  = str(row.get("location", "") or "")
            locations = _infer_preferred_locations(location, budget, df_jobs)
            profiles.append(
                FreelancerProfile(
                    freelancer_id          = str(row.get("id", "")),
                    embedding              = embeddings[i].astype(np.float32),
                    preferred_budget_range = budget,
                    preferred_locations    = locations,
                )
            )
        return profiles

    def _load_cache(self) -> dict | None:
        if not self.cache_path.exists():
            return None
        try:
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)
            logger.info(
                "Loaded cache from %s (%d freelancers, %d jobs)",
                self.cache_path,
                len(cache.get("freelancer_ids", [])),
                len(cache.get("job_ids", [])),
            )
            return cache
        except Exception as exc:
            logger.warning("Cache load failed (%s) — recomputing.", exc)
            return None

    def _save_cache(self, cache: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Embeddings cached at %s", self.cache_path)
