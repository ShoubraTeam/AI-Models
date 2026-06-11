# ---------------------------------------------------------------
# Recommendation System Agent
# Wraps BAAI/bge-base-en-v1.5 SentenceTransformer
# ---------------------------------------------------------------

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class RSEmbeddingEngine:
    """
    Lightweight embedding engine for the recommendation system.
    Encodes a single cleaned text string and returns its dense
    float32 embedding vector as a plain Python list (JSON-ready).
    """

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def embed(self, text: str) -> list[float]:
        vector: np.ndarray = self._model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vector.tolist()