"""
EmbeddingEngine
===============
Wraps SentenceTransformer encoding with:
  - Incremental update  (only re-encode new IDs)
  - Pickle cache        (load_or_compute avoids redundant GPU work)
  - Model benchmarking  (compare multiple models on a sample)
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from config.settings import (
    EMBEDDING_MODEL, BATCH_SIZE, EMBEDDING_DIM, EMBEDDINGS_CACHE,
)

logger = logging.getLogger(__name__)


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
        return {
            "freelancer_embeddings": f_emb,
            "job_embeddings":        j_emb,
            "freelancer_ids":        df_freelancers["id"].tolist(),
            "job_ids":               df_jobs["projectId"].tolist(),
            "model_name":            self.model_name,
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
