"""
FastAPI serving layer
=====================
Exposes the RecommendationEngine as a REST API.

Endpoints
---------
GET  /v1/health
POST /v1/recommend/jobs                        → freelancer → top-N jobs  (internal ranking)
POST /v1/recommend/freelancers                 → job → top-N freelancers  (internal ranking)
GET  /v1/freelancers/{freelancer_id}           → raw freelancer record
GET  /v1/freelancers/{freelancer_id}/profile   → embedding + ALL scoring signals (backend does ranking)
GET  /v1/job-embeddings                   → ALL job embeddings + scoring signals (backend stores & ranks)
GET  /v1/job-embeddings/{job_index}            → single job embedding + scoring signals
GET  /v1/jobs/{job_index}                      → raw job record

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import math
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import (
    API_VERSION, DEFAULT_TOP_N, MIN_SCORE_THRESHOLD,
    WEIGHT_SEMANTIC, WEIGHT_STRUCTURED,
    WEIGHT_RATE_COMPAT, WEIGHT_REPUTATION,
    GEO_BONUS, WEEKLY_HOURS,
)
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


# ── Scoring config (returned with every profile so backend can replicate) ─────
SCORING_CONFIG = {
    "weight_semantic":    WEIGHT_SEMANTIC,
    "weight_structured":  WEIGHT_STRUCTURED,
    "weight_rate_compat": WEIGHT_RATE_COMPAT,
    "weight_reputation":  WEIGHT_REPUTATION,
    "geo_bonus":          GEO_BONUS,
    "weekly_hours":       WEEKLY_HOURS,
    "formula":            (
        "hybrid = weight_semantic * cosine_similarity(f_emb, j_emb)"
        " + weight_structured * (weight_rate_compat * rate_compat"
        "                       + weight_reputation * reputation)"
        " + (geo_bonus if f.country_code == j.country_code else 0)"
        " — clipped to 1.0"
    ),
    "rate_compat_formula": (
        "implied_rate = job.budget_avg / weekly_hours; "
        "ratio = freelancer.rate_usd / implied_rate; "
        "rate_compat = exp(-0.5 * (ln(ratio))^2 / 0.64)"
    ),
    "reputation_formula": (
        "jobs_norm = min(freelancer.jobs_done / 50, 1.0); "
        "reputation = 0.7 * freelancer.feedback_score + 0.3 * jobs_norm"
    ),
}

# ── Budget bucket helper ───────────────────────────────────────────────────────
_BUDGET_BINS   = [0, 50, 250, 1_000, 5_000, float("inf")]
_BUDGET_LABELS = ["micro", "small", "medium", "large", "enterprise"]

def _budget_label(avg: float) -> str:
    for lo, hi, label in zip(_BUDGET_BINS, _BUDGET_BINS[1:], _BUDGET_LABELS):
        if lo <= avg < hi:
            return label
    return "enterprise"

def _is_nan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False

def _safe_float(v: Any, default: float = 0.0) -> float:
    """Convert to float safely, returning default for None/NaN/inf."""
    if v is None:
        return default
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default

def _safe_int(v: Any, default: int = 0) -> int:
    """Convert to int safely, returning default for None/NaN."""
    try:
        f = float(v)
        return default if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return default

def _safe_str(v: Any) -> str:
    """Convert to str safely, returning empty string for None/NaN/null."""
    if v is None:
        return ""
    s = str(v)
    return "" if s.lower() in ("nan", "none", "null") else s


# ── Schemas ───────────────────────────────────────────────────────────────────

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


class FreelancerScoringSignals(BaseModel):
    """
    All freelancer-side signals needed to compute the hybrid score on the backend.

    rate_compat  = exp(-0.5 * (ln(rate_usd / (job.budget_avg / weekly_hours)))^2 / 0.64)
    reputation   = 0.7 * feedback_score + 0.3 * min(jobs_done / 50, 1.0)
    geo_bonus    = scoring_config.geo_bonus  if  country_code == job.country_code
    hybrid       = w_semantic * cosine_sim + w_structured * (w_rate * rate_compat + w_rep * reputation) + geo_bonus
    """
    rate_usd:         Optional[float] = Field(None,  description="Hourly rate in USD")
    feedback_score:   float           = Field(...,   description="Feedback percentage normalised [0, 1]")
    jobs_done:        int             = Field(...,   description="Completed fixed-price jobs count")
    country_code:     str             = Field(...,   description="ISO-2 country code for geo bonus")
    reputation_score: float           = Field(...,   description="Pre-computed: 0.7*feedback + 0.3*min(jobs_done/50,1)")


class FreelancerProfileResponse(BaseModel):
    """
    Full freelancer profile — store once, use for self-managed ranking.

    Backend workflow
    ----------------
    1. Fetch + store this once per freelancer (refresh on profile update).
    2. Fetch + store all jobs via GET /v1/job-embeddings.
    3. At recommendation time:
       a. Filter jobs:  budget_label IN preferred_budget_range
                        AND client_country IN preferred_locations
       b. Rank filtered jobs:
            similarity     = cosine_similarity(this.embedding, job.embedding)
            rate_compat    = exp(-0.5 * ln(rate_usd / (job.budget_avg / weekly_hours))^2 / 0.64)
            reputation     = scoring_signals.reputation_score
            structured     = w_rate * rate_compat + w_rep * reputation
            geo            = geo_bonus if scoring_signals.country_code == job.country_code else 0
            hybrid_score   = w_sem * similarity + w_struct * structured + geo   (clip to 1.0)
       c. Return top-N by hybrid_score.
    """
    freelancer_id:          str
    embedding:              list[float]         = Field(..., description="Semantic vector for cosine similarity")
    preferred_budget_range: tuple[str, str]     = Field(..., description="(min_label, max_label) — use as budget pre-filter")
    preferred_locations:    list[str]           = Field(..., description="Preferred client countries — use as geo pre-filter")
    scoring_signals:        FreelancerScoringSignals
    scoring_config:         dict[str, Any]      = Field(..., description="Exact weights + formulas to replicate hybrid scoring")


class JobEmbeddingResponse(BaseModel):
    """
    Job embedding + all filter and scoring signals.

    Filter fields (narrow candidates before similarity):
      budget_label    → compare against freelancer.preferred_budget_range
      client_country  → compare against freelancer.preferred_locations

    Scoring fields (re-rank after similarity):
      budget_avg      → implied_rate = budget_avg / weekly_hours  → rate_compat
      country_code    → compare to freelancer.country_code for geo bonus
      client_rating   → optional quality signal
      review_count    → optional quality signal
    """
    job_index:      int
    job_id:         Optional[int]
    job_title:      str
    embedding:      list[float] = Field(..., description="Semantic vector for cosine similarity")
    budget_min:     float       = Field(..., description="Min posted budget (USD)")
    budget_max:     float       = Field(..., description="Max posted budget (USD)")
    budget_avg:     float       = Field(..., description="Avg budget (USD) — used in rate_compat formula")
    budget_label:   str         = Field(..., description="Budget tier: micro|small|medium|large|enterprise")
    client_country: str         = Field(..., description="Client country name — filter against preferred_locations")
    client_state:   str         = Field(..., description="Client state/city")
    country_code:   str         = Field(..., description="ISO-2 code — compare to freelancer.country_code for geo bonus")
    client_rating:  float       = Field(..., description="Client average rating (0–5)")
    review_count:   int         = Field(..., description="Number of client reviews")
    tags:           str         = Field(..., description="Cleaned skill tags")
    scoring_config: dict[str, Any] = Field(..., description="Weights + formulas — identical for all jobs")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(f"/{API_VERSION}/health")
async def health():
    return {
        "status": "ok",
        "engine_ready": engine is not None,
        "freelancer_count": len(engine.df_freelancers) if engine is not None else 0,
        "job_count": len(engine.df_jobs) if engine is not None else 0,
    }


# ── Internal ranking (API does everything) ────────────────────────────────────

@app.post(f"/{API_VERSION}/recommend/jobs", response_model=list[MatchResponse])
async def recommend_jobs(req: JobRecommendRequest):
    """Top-N jobs for a freelancer — full ranking done inside the API."""
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
    """Top-N freelancers for a job — full ranking done inside the API."""
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


# ── Freelancer profile (backend does ranking) ─────────────────────────────────

@app.get(f"/{API_VERSION}/freelancers/{{freelancer_id}}")
async def get_freelancer(freelancer_id: str):
    """Raw freelancer record lookup."""
    _check_engine()
    row = engine.df_freelancers[engine.df_freelancers["id"] == freelancer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Freelancer '{freelancer_id}' not found.")
    return row.iloc[0].to_dict()


@app.get(
    f"/{API_VERSION}/freelancers/{{freelancer_id}}/profile",
    response_model=FreelancerProfileResponse,
)
async def get_freelancer_profile(freelancer_id: str):
    """
    Full freelancer profile for backend storage and self-managed ranking.

    Returns embedding + preferred_budget_range + preferred_locations
    + all scoring signals + exact formulas.
    Backend stores this and uses it to rank jobs without calling back.
    """
    _check_engine()
    try:
        profile = engine.get_freelancer_profile(freelancer_id=freelancer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Freelancer '{freelancer_id}' not found.")
    except IndexError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    row = engine.df_freelancers[engine.df_freelancers["id"] == freelancer_id].iloc[0]

    feedback   = _safe_float(row.get("feedback_score"))
    jobs_done  = _safe_int(row.get("jobs_done"))
    jobs_norm  = min(jobs_done / 50, 1.0)
    reputation = round(0.7 * feedback + 0.3 * jobs_norm, 4)

    rate     = row.get("rate_usd")
    rate_val = float(rate) if rate is not None and not _is_nan(rate) else None

    signals = FreelancerScoringSignals(
        rate_usd         = round(rate_val, 4) if rate_val is not None else None,
        feedback_score   = round(feedback, 4),
        jobs_done        = jobs_done,
        country_code     = _safe_str(row.get("country_code")),
        reputation_score = reputation,
    )

    return FreelancerProfileResponse(
        freelancer_id          = profile.freelancer_id,
        embedding              = [round(float(x), 6) for x in profile.embedding],
        preferred_budget_range = profile.preferred_budget_range,
        preferred_locations    = profile.preferred_locations,
        scoring_signals        = signals,
        scoring_config         = SCORING_CONFIG,
    )


# ── Job embeddings (backend does ranking) ─────────────────────────────────────

@app.get(
    f"/{API_VERSION}/job-embeddings",
    response_model=list[JobEmbeddingResponse],
)
async def get_all_job_embeddings(
    offset: int = Query(0,    ge=0,         description="Pagination start index"),
    limit:  int = Query(1000, ge=1, le=5000, description="Max records per page"),
):
    """
    All job embeddings + filter and scoring signals — paginated.

    Call repeatedly to populate the backend job store:
      GET /v1/job-embeddings?offset=0&limit=1000
      GET /v1/job-embeddings?offset=1000&limit=1000
      ... until response length < limit

    Each record contains everything needed to filter and score
    without further API calls at recommendation time.
    """
    _check_engine()

    df    = engine.df_jobs
    emb   = engine._j_emb
    total = len(df)
    end   = min(offset + limit, total)

    if offset >= total:
        return []

    result = []
    for i in range(offset, end):
        row = df.iloc[i]
        vec = emb[i]

        budget_avg    = _safe_float(row.get("budget_avg"))
        budget_min    = _safe_float(row.get("budget_min"))
        budget_max    = _safe_float(row.get("budget_max"))
        client_rating = _safe_float(row.get("client_rating"))
        review_count  = _safe_int(row.get("review_count"))

        result.append(JobEmbeddingResponse(
            job_index      = i,
            job_id         = _safe_int(row.get("projectId"), default=-1) if row.get("projectId") is not None else None,
            job_title      = _safe_str(row.get("job_title")),
            embedding      = [round(float(x), 6) for x in vec],
            budget_min     = round(budget_min, 2),
            budget_max     = round(budget_max, 2),
            budget_avg     = round(budget_avg, 2),
            budget_label   = _budget_label(budget_avg),
            client_country = _safe_str(row.get("client_country")),
            client_state   = _safe_str(row.get("client_state")),
            country_code   = _safe_str(row.get("country_code")),
            client_rating  = round(client_rating, 2),
            review_count   = review_count,
            tags           = _safe_str(row.get("tags_cleaned")),
            scoring_config = SCORING_CONFIG,
        ))

    return result


@app.get(
    f"/{API_VERSION}/job-embeddings/{{job_index}}",
    response_model=JobEmbeddingResponse,
)
async def get_job_embedding(job_index: int):
    """
    Single job embedding + filter and scoring signals by index.
    Use to refresh one job in the backend store without re-fetching everything.
    """
    _check_engine()

    if job_index < 0 or job_index >= len(engine.df_jobs):
        raise HTTPException(status_code=404, detail=f"Job index {job_index} out of range.")

    row = engine.df_jobs.iloc[job_index]
    vec = engine._j_emb[job_index]

    budget_avg    = _safe_float(row.get("budget_avg"))
    budget_min    = _safe_float(row.get("budget_min"))
    budget_max    = _safe_float(row.get("budget_max"))
    client_rating = _safe_float(row.get("client_rating"))
    review_count  = _safe_int(row.get("review_count"))

    return JobEmbeddingResponse(
        job_index      = job_index,
        job_id         = _safe_int(row.get("projectId"), default=-1) if row.get("projectId") is not None else None,
        job_title      = _safe_str(row.get("job_title")),
        embedding      = [round(float(x), 6) for x in vec],
        budget_min     = round(budget_min, 2),
        budget_max     = round(budget_max, 2),
        budget_avg     = round(budget_avg, 2),
        budget_label   = _budget_label(budget_avg),
        client_country = _safe_str(row.get("client_country")),
        client_state   = _safe_str(row.get("client_state")),
        country_code   = _safe_str(row.get("country_code")),
        client_rating  = round(client_rating, 2),
        review_count   = review_count,
        tags           = _safe_str(row.get("tags_cleaned")),
        scoring_config = SCORING_CONFIG,
    )


@app.get(f"/{API_VERSION}/jobs/{{job_index}}")
async def get_job(job_index: int):
    """Raw job record lookup by index."""
    _check_engine()
    if job_index < 0 or job_index >= len(engine.df_jobs):
        raise HTTPException(status_code=404, detail=f"Job index {job_index} out of range.")
    return engine.df_jobs.iloc[job_index].to_dict()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_engine() -> None:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")
