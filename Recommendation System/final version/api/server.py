"""
FastAPI serving layer
=====================
Exposes the RecommendationEngine as a REST API.

Endpoints
---------
GET  /v1/health                          → liveness check
POST /v1/recommend/jobs                  → freelancer → top-N jobs
POST /v1/recommend/freelancers           → job → top-N freelancers
GET  /v1/freelancers/{freelancer_id}     → freelancer profile lookup
GET  /v1/jobs/{job_index}               → job record lookup

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import API_VERSION, DEFAULT_TOP_N, MIN_SCORE_THRESHOLD
from models.recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)

# ── Global engine (loaded once at startup) ────────────────────────────────────
engine: RecommendationEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # If the engine was pre-built and injected by main.py --serve, skip rebuild.
    if engine is None:
        logger.info("Starting up — building RecommendationEngine …")
        engine = RecommendationEngine()
        engine.build()
        logger.info("Engine ready.")
    else:
        logger.info("Starting up — using pre-built RecommendationEngine.")
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


# ── Request / response schemas ────────────────────────────────────────────────

class JobRecommendRequest(BaseModel):
    freelancer_id: str | None = Field(None, description="Freelancer UUID string")
    freelancer_index: int | None = Field(None, description="Row index (0-based)")
    top_n: int = Field(DEFAULT_TOP_N, ge=1, le=100)
    min_score: float = Field(MIN_SCORE_THRESHOLD, ge=0.0, le=1.0)


class FreelancerRecommendRequest(BaseModel):
    job_index: int = Field(..., description="Job row index (0-based)")
    top_n: int = Field(DEFAULT_TOP_N, ge=1, le=100)
    min_score: float = Field(MIN_SCORE_THRESHOLD, ge=0.0, le=1.0)


class MatchResponse(BaseModel):
    rank: int
    job_title: str
    semantic_score: float
    structured_score: float
    hybrid_score: float
    geo_bonus_applied: bool
    payload: dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Redirect browsers and health probes to the API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get(f"/{API_VERSION}/health")
async def health():
    return {"status": "ok", "engine_ready": engine is not None}


@app.get(f"/{API_VERSION}/recommend/jobs")
async def recommend_jobs_method_hint():
    """Friendly error for clients that accidentally use GET instead of POST."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=405,
        content={
            "detail": (
                "This endpoint requires a POST request with a JSON body. "
                "Example: POST /v1/recommend/jobs  "
                "Body: {\"freelancer_index\": 0, \"top_n\": 5}"
            )
        },
    )


@app.get(f"/{API_VERSION}/recommend/freelancers")
async def recommend_freelancers_method_hint():
    """Friendly error for clients that accidentally use GET instead of POST."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=405,
        content={
            "detail": (
                "This endpoint requires a POST request with a JSON body. "
                "Example: POST /v1/recommend/freelancers  "
                "Body: {\"job_index\": 0, \"top_n\": 5}"
            )
        },
    )


@app.post(f"/{API_VERSION}/recommend/jobs", response_model=list[MatchResponse])
async def recommend_jobs(req: JobRecommendRequest):
    """Return top-N jobs for a given freelancer."""
    _check_engine()
    try:
        matches = engine.recommend_jobs(
            freelancer_id=req.freelancer_id,
            freelancer_index=req.freelancer_index,
            top_n=req.top_n,
            min_score=req.min_score,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return [
        MatchResponse(
            rank=i + 1,
            job_title=m.job_title,
            semantic_score=round(m.semantic_score, 4),
            structured_score=round(m.structured_score, 4),
            hybrid_score=round(m.hybrid_score, 4),
            geo_bonus_applied=m.geo_bonus_applied,
            payload=m.payload,
        )
        for i, m in enumerate(matches)
    ]


@app.post(f"/{API_VERSION}/recommend/freelancers", response_model=list[MatchResponse])
async def recommend_freelancers(req: FreelancerRecommendRequest):
    """Return top-N freelancers for a given job (reverse matching)."""
    _check_engine()
    try:
        matches = engine.recommend_freelancers(
            job_index=req.job_index,
            top_n=req.top_n,
            min_score=req.min_score,
        )
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Job index {req.job_index} out of range.")

    return [
        MatchResponse(
            rank=i + 1,
            job_title=m.job_title,
            semantic_score=round(m.semantic_score, 4),
            structured_score=round(m.structured_score, 4),
            hybrid_score=round(m.hybrid_score, 4),
            geo_bonus_applied=m.geo_bonus_applied,
            payload=m.payload,
        )
        for i, m in enumerate(matches)
    ]


@app.get(f"/{API_VERSION}/freelancers/{{freelancer_id}}")
async def get_freelancer(freelancer_id: str):
    """Look up a freelancer profile by ID."""
    _check_engine()
    row = engine.df_freelancers[engine.df_freelancers["id"] == freelancer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Freelancer '{freelancer_id}' not found.")
    return row.iloc[0].to_dict()


@app.get(f"/{API_VERSION}/jobs/{{job_index}}")
async def get_job(job_index: int):
    """Look up a job by row index."""
    _check_engine()
    if job_index < 0 or job_index >= len(engine.df_jobs):
        raise HTTPException(status_code=404, detail=f"Job index {job_index} out of range.")
    return engine.df_jobs.iloc[job_index].to_dict()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_engine() -> None:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")