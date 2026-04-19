"""
VectorStore
===========
Thin wrapper around Qdrant that replaces the brute-force
cosine_similarity(all_freelancers, all_jobs) approach with
approximate nearest-neighbour (ANN) HNSW search.

Complexity: O(n) → O(log n) per query.

Compatibility
-------------
qdrant-client ≥ 1.7 removed the legacy `.search()` method and replaced
it with `.query_points()`.  This module detects which API is available
at import time and routes calls accordingly, so the code works on both
old and new installations without any manual changes.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from config.settings import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_JOBS_COLLECTION,
    QDRANT_FREELANCERS_COLLECTION,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)

# ── Detect which search API this qdrant-client version exposes ────────────────
_USE_QUERY_POINTS = hasattr(QdrantClient, "query_points")
logger.debug(
    "qdrant-client search API: %s",
    "query_points (>=1.7)" if _USE_QUERY_POINTS else "search (legacy)",
)


@dataclass
class ScoredPoint:
    """
    Minimal scored-point object returned by both search paths.

    The legacy API returns qdrant_client.models.ScoredPoint directly;
    the new query_points API wraps results in a QueryResponse.  We
    normalise both into this lightweight dataclass so the rest of the
    codebase never has to care about the version difference.
    """
    id: int
    score: float
    payload: dict[str, Any]


class VectorStore:
    """Manage Qdrant collections for jobs and freelancers."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
    ) -> None:
        if host == ":memory:":
            self._client = QdrantClient(location=":memory:")
            logger.info("Qdrant running in-memory.")
        else:
            self._client = QdrantClient(host=host, port=port)
            logger.info("Qdrant connected at %s:%d", host, port)

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_jobs(
        self,
        embeddings: np.ndarray,
        df_jobs: pd.DataFrame,
        recreate: bool = False,
    ) -> None:
        """Upload job embeddings + metadata into Qdrant."""
        self._ensure_collection(QDRANT_JOBS_COLLECTION, recreate)
        points = [
            PointStruct(
                id=int(i),
                vector=embeddings[i].tolist(),
                payload=self._safe_payload(df_jobs.iloc[i]),
            )
            for i in range(len(embeddings))
        ]
        self._upsert_batched(QDRANT_JOBS_COLLECTION, points)
        logger.info("Indexed %d jobs into '%s'", len(points), QDRANT_JOBS_COLLECTION)

    def index_freelancers(
        self,
        embeddings: np.ndarray,
        df_freelancers: pd.DataFrame,
        recreate: bool = False,
    ) -> None:
        """Upload freelancer embeddings + metadata into Qdrant."""
        self._ensure_collection(QDRANT_FREELANCERS_COLLECTION, recreate)
        points = [
            PointStruct(
                id=int(i),
                vector=embeddings[i].tolist(),
                payload=self._safe_payload(df_freelancers.iloc[i]),
            )
            for i in range(len(embeddings))
        ]
        self._upsert_batched(QDRANT_FREELANCERS_COLLECTION, points)
        logger.info(
            "Indexed %d freelancers into '%s'",
            len(points),
            QDRANT_FREELANCERS_COLLECTION,
        )

    # ── Search ────────────────────────────────────────────────────────────

    def search_jobs(
        self,
        query_vector: np.ndarray | list[float],
        top_n: int = 10,
        score_threshold: float = 0.0,
    ) -> list[ScoredPoint]:
        """Return top-N job matches for a freelancer query vector."""
        return self._query(
            QDRANT_JOBS_COLLECTION,
            self._to_list(query_vector),
            top_n,
            score_threshold,
        )

    def search_freelancers(
        self,
        query_vector: np.ndarray | list[float],
        top_n: int = 10,
        score_threshold: float = 0.0,
    ) -> list[ScoredPoint]:
        """Return top-N freelancer matches for a job query vector (reverse)."""
        return self._query(
            QDRANT_FREELANCERS_COLLECTION,
            self._to_list(query_vector),
            top_n,
            score_threshold,
        )

    def collection_exists(self, name: str) -> bool:
        try:
            self._client.get_collection(name)
            return True
        except Exception:
            return False

    # ── Internals ─────────────────────────────────────────────────────────

    def _query(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        score_threshold: float,
    ) -> list[ScoredPoint]:
        """
        Version-safe ANN search.

        * qdrant-client >= 1.7  →  query_points()
        * qdrant-client <  1.7  →  search()  (legacy)

        Both paths return a list of our local ScoredPoint dataclass.
        """
        if _USE_QUERY_POINTS:
            # New API (>= 1.7): query_points accepts a plain list as query.
            # Returns a QueryResponse; its .points holds the scored results.
            response = self._client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                score_threshold=score_threshold if score_threshold > 0 else None,
            )
            raw_points = response.points
        else:
            # Legacy API
            raw_points = self._client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                score_threshold=score_threshold,
            )

        # Normalise to our own ScoredPoint so downstream code is insulated
        # from qdrant_client internal type changes.
        return [
            ScoredPoint(
                id=int(p.id),
                score=float(p.score),
                payload=p.payload or {},
            )
            for p in raw_points
        ]

    def _ensure_collection(self, name: str, recreate: bool) -> None:
        if recreate and self.collection_exists(name):
            self._client.delete_collection(name)
            logger.info("Deleted existing collection '%s'", name)

        if not self.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )
            logger.info("Created collection '%s'", name)

    def _upsert_batched(
        self, collection: str, points: list[PointStruct], batch: int = 256
    ) -> None:
        for i in range(0, len(points), batch):
            self._client.upsert(collection, points[i : i + batch])

    @staticmethod
    def _safe_payload(row: pd.Series) -> dict[str, Any]:
        """Convert a DataFrame row to a JSON-safe dict."""
        payload = {}
        for k, v in row.items():
            if isinstance(v, float) and np.isnan(v):
                payload[k] = None
            elif isinstance(v, (np.integer,)):
                payload[k] = int(v)
            elif isinstance(v, (np.floating,)):
                payload[k] = float(v)
            else:
                payload[k] = v
        return payload

    @staticmethod
    def _to_list(v: np.ndarray | list) -> list[float]:
        return v.tolist() if isinstance(v, np.ndarray) else v
