"""
FastAPI serving layer
=====================
Exposes two embedding endpoints.

Endpoints
---------
GET  /v1/health
POST /v1/freelancers/embed   → preprocess freelancer (bio + skills) → embedding
POST /v1/jobs/embed          → preprocess job (title + description + skills) → embedding

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import API_VERSION
from models.recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)

# ── Global engine ─────────────────────────────────────────────────────────────
engine: RecommendationEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Starting up — building RecommendationEngine ...")
    engine = RecommendationEngine()
    engine.build()
    logger.info("Engine ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Freelancer Job Recommender API",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Input schemas ─────────────────────────────────────────────────────────────

class FreelancerInput(BaseModel):
    """Raw freelancer data: bio and skills."""
    bio:    Optional[str] = Field(None, description="Freelancer bio / profile description")
    skills: Optional[str] = Field(None, description="Comma-separated skills or raw skill string")


class JobInput(BaseModel):
    """Raw job data: title, description, and skills list."""
    job_title:       Optional[str]       = Field(None, description="Job title")
    job_description: Optional[str]       = Field(None, description="Full job description text")
    skills:          Optional[list[str]] = Field(None, description="List of required skills/tags")


# ── Output schema ─────────────────────────────────────────────────────────────

class EmbeddingResponse(BaseModel):
    """Embedding vector produced after preprocessing the input."""
    embedding: list[float] = Field(..., description="Semantic vector for cosine similarity")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(f"/{API_VERSION}/health")
async def health():
    return {
        "status": "ok",
        "engine_ready": engine is not None,
    }


@app.post(
    f"/{API_VERSION}/freelancers/embed",
    response_model=EmbeddingResponse,
)
async def embed_freelancer(req: FreelancerInput):
    """
    Preprocess freelancer data and return its embedding.

    Steps:
      1. Clean and combine bio + skills into enriched text
      2. Generate semantic embedding via the ML model
      3. Return the embedding vector
    """
    _check_engine()

    preprocessor = engine._preprocessor
    embedder     = engine._embedder

    # Map to preprocessor's expected keys
    input_data = {
        "description": req.bio or "",
        "skills":      req.skills or "",
    }

    row = preprocessor.process_freelancer_input(input_data)
    emb = embedder.encode([row["enriched_text"]], desc="Freelancer embed")[0]

    return EmbeddingResponse(
        embedding=[round(float(x), 6) for x in emb],
    )


@app.post(
    f"/{API_VERSION}/jobs/embed",
    response_model=EmbeddingResponse,
)
async def embed_job(req: JobInput):
    """
    Preprocess job data and return its embedding.

    Steps:
      1. Clean skills list, build enriched text from title + description + skills
      2. Generate semantic embedding via the ML model
      3. Return the embedding vector
    """
    _check_engine()

    preprocessor = engine._preprocessor
    embedder     = engine._embedder

    input_data = {
        "job_title":       req.job_title or "",
        "job_description": req.job_description or "",
        "tags":            req.skills or [],
    }

    row = preprocessor.process_job_input(input_data)
    emb = embedder.encode([row["enriched_text"]], desc="Job embed")[0]

    return EmbeddingResponse(
        embedding=[round(float(x), 6) for x in emb],
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_engine() -> None:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")
