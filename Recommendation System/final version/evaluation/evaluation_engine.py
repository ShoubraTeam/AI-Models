"""
EvaluationEngine
================
Proper information-retrieval evaluation replacing the original
"mean cosine similarity" proxy.

Metrics
-------
- NDCG@K       Normalised Discounted Cumulative Gain
- Precision@K  Fraction of top-K results that are relevant
- MeanSim@K    Original metric, kept for backwards comparison

Relevance proxy
---------------
Since no click / hire labels exist, Jaccard overlap between
freelancer skills and job tags is used as a continuous relevance
signal.  This is a reasonable proxy and can be swapped for real
labels when available.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import (
    EVAL_TOP_N,
    EVAL_SAMPLE_FREELANCERS,
    EVAL_SAMPLE_JOBS,
)

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Evaluate recommendation quality with proper IR metrics."""

    def __init__(
        self,
        df_freelancers: pd.DataFrame,
        df_jobs: pd.DataFrame,
        freelancer_embeddings: np.ndarray,
        job_embeddings: np.ndarray,
    ) -> None:
        self.df_f  = df_freelancers
        self.df_j  = df_jobs
        self.f_emb = freelancer_embeddings
        self.j_emb = job_embeddings

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        top_n: int = EVAL_TOP_N,
        sample_freelancers: int = EVAL_SAMPLE_FREELANCERS,
        sample_jobs: int = EVAL_SAMPLE_JOBS,
    ) -> pd.DataFrame:
        """
        Compute NDCG@K, Precision@K, and MeanSim@K over a random sample.
        Returns a summary DataFrame.
        """
        logger.info(
            "Evaluating on %d freelancers × %d jobs, top_n=%d",
            sample_freelancers, sample_jobs, top_n,
        )

        f_idx = np.random.choice(len(self.f_emb), min(sample_freelancers, len(self.f_emb)), replace=False)
        j_idx = np.random.choice(len(self.j_emb), min(sample_jobs, len(self.j_emb)), replace=False)

        f_sample = self.f_emb[f_idx]
        j_sample = self.j_emb[j_idx]
        df_f_s   = self.df_f.iloc[f_idx].reset_index(drop=True)
        df_j_s   = self.df_j.iloc[j_idx].reset_index(drop=True)

        # Full similarity matrix for the sample
        sims = cosine_similarity(f_sample, j_sample)   # (n_f, n_j)

        ndcg_scores  = []
        prec_scores  = []
        mean_sims    = []

        for i in range(len(f_sample)):
            relevance = self._relevance_row(df_f_s.iloc[i], df_j_s)
            sim_row   = sims[i]

            # NDCG@K
            ndcg_scores.append(ndcg_score([relevance], [sim_row], k=top_n))

            # Precision@K  (relevant = Jaccard > median)
            threshold = np.median(relevance)
            top_idx   = np.argsort(sim_row)[::-1][:top_n]
            hits      = sum(1 for j in top_idx if relevance[j] > threshold)
            prec_scores.append(hits / top_n)

            # MeanSim@K (original metric for comparison)
            top_sims  = np.sort(sim_row)[::-1][:top_n]
            mean_sims.append(np.mean(top_sims))

        summary = pd.DataFrame({
            "metric": [f"NDCG@{top_n}", f"Precision@{top_n}", f"MeanSim@{top_n}"],
            "mean":   [np.mean(ndcg_scores), np.mean(prec_scores), np.mean(mean_sims)],
            "std":    [np.std(ndcg_scores),  np.std(prec_scores),  np.std(mean_sims)],
            "median": [np.median(ndcg_scores), np.median(prec_scores), np.median(mean_sims)],
        })
        summary["mean"]   = summary["mean"].round(4)
        summary["std"]    = summary["std"].round(4)
        summary["median"] = summary["median"].round(4)

        self._log_summary(summary)
        self._all_ndcg = ndcg_scores   # kept for plot_distribution
        self._all_prec = prec_scores
        return summary

    def plot_distribution(
        self,
        top_n: int = EVAL_TOP_N,
        save_path: str | None = None,
    ) -> None:
        """
        Plot NDCG and Precision distributions side-by-side.
        Requires evaluate() to have been called first.
        """
        if not hasattr(self, "_all_ndcg"):
            raise RuntimeError("Call evaluate() before plot_distribution().")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Recommendation Quality — top-{top_n}", fontsize=13)

        axes[0].hist(self._all_ndcg, bins=30, color="#5DCAA5", edgecolor="white")
        axes[0].set_title(f"NDCG@{top_n} distribution")
        axes[0].set_xlabel("NDCG score")
        axes[0].set_ylabel("Frequency")
        axes[0].axvline(np.mean(self._all_ndcg), color="#0F6E56", linestyle="--",
                        label=f"Mean = {np.mean(self._all_ndcg):.3f}")
        axes[0].legend()

        axes[1].hist(self._all_prec, bins=20, color="#7F77DD", edgecolor="white")
        axes[1].set_title(f"Precision@{top_n} distribution")
        axes[1].set_xlabel("Precision score")
        axes[1].set_ylabel("Frequency")
        axes[1].axvline(np.mean(self._all_prec), color="#3C3489", linestyle="--",
                        label=f"Mean = {np.mean(self._all_prec):.3f}")
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Plot saved to %s", save_path)
        plt.show()

    def compare_models(
        self,
        model_benchmark_df: pd.DataFrame,
    ) -> None:
        """Visualise model benchmark results from EmbeddingEngine.benchmark_models()."""
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#5DCAA5" if i == 0 else "#B4B2A9"
                  for i in range(len(model_benchmark_df))]
        ax.barh(model_benchmark_df["model"], model_benchmark_df["mean_top5_sim"],
                color=colors, edgecolor="white")
        ax.set_xlabel("Mean top-5 cosine similarity")
        ax.set_title("Embedding model comparison")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    # ── Relevance proxy ───────────────────────────────────────────────────

    @staticmethod
    def _relevance_row(freelancer: pd.Series, df_jobs: pd.DataFrame) -> np.ndarray:
        """
        Jaccard similarity between freelancer skills and each job's tags.
        Returns a float array of shape (n_jobs,).
        """
        f_skills = set(str(freelancer.get("skills_cleaned") or "").lower().split(", "))
        f_skills.discard("")

        def jaccard(tags_str: object) -> float:
            j_tags = set(str(tags_str or "").lower().split(", "))
            j_tags.discard("")
            if not f_skills and not j_tags:
                return 0.0
            return len(f_skills & j_tags) / len(f_skills | j_tags)

        return df_jobs["tags_cleaned"].apply(jaccard).values

    # ── Logging ───────────────────────────────────────────────────────────

    @staticmethod
    def _log_summary(df: pd.DataFrame) -> None:
        logger.info("\n%s", df.to_string(index=False))
        print("\n" + "="*50)
        print("  Evaluation Summary")
        print("="*50)
        print(df.to_string(index=False))
        print()
