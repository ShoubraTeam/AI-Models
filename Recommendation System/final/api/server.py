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

from fastapi import FastAPI, HTTPException
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


# ── Input schemas (data sent BY the backend) ─────────────────────────────────

class FreelancerInput(BaseModel):
    """
    Raw freelancer data sent by the backend.
    The API processes this, embeds it, and returns the full profile.
    All fields are optional — missing fields get sensible defaults.
    """
    freelancer_id:    str           = Field(...,  description="Your internal freelancer ID")
    job_title:        Optional[str] = Field(None, description="Freelancer's headline / job title")
    skills:           Optional[str] = Field(None, description="Comma-separated skills or raw skill string")
    description:      Optional[str] = Field(None, description="Freelancer bio / profile description")
    hour_rate:        Optional[str] = Field(None, description="Hourly rate as string e.g. '$15' or '15'")
    feedback_percent: Optional[str] = Field(None, description="Job success score e.g. '98%' or '0.98'")
    fixed_jobs_done:  Optional[str] = Field(None, description="Completed jobs e.g. '12 fixed price jobs' or '12'")
    location:         Optional[str] = Field(None, description="Country name e.g. 'Pakistan'")


class JobInput(BaseModel):
    """
    Raw job data sent by the backend.
    The API processes this, embeds it, and returns the full job profile.
    All fields are optional — missing fields get sensible defaults.
    """
    job_id:                   Optional[Any] = Field(None, description="Your internal job ID")
    job_title:                Optional[str] = Field(None, description="Job title")
    job_description:          Optional[str] = Field(None, description="Full job description text")
    tags:                     Optional[Any] = Field(None, description="Skills/tags — string or list")
    client_country:           Optional[str] = Field(None, description="Client country name e.g. 'Germany'")
    client_state:             Optional[str] = Field(None, description="Client state or city")
    client_average_rating:    Optional[float] = Field(None, description="Client rating 0–5")
    client_review_count:      Optional[int]   = Field(None, description="Number of client reviews")
    min_price:                Optional[float] = Field(None, description="Minimum budget (USD)")
    max_price:                Optional[float] = Field(None, description="Maximum budget (USD)")
    avg_price:                Optional[float] = Field(None, description="Average budget (USD)")


# ── Output schemas (returned TO the backend) ─────────────────────────────────

class FreelancerScoringSignals(BaseModel):
    """
    All freelancer-side signals the backend needs to compute hybrid scores.

    rate_compat  = exp(-0.5 * (ln(rate_usd / (job.budget_avg / weekly_hours)))^2 / 0.64)
    reputation   = 0.7 * feedback_score + 0.3 * min(jobs_done / 50, 1.0)
    geo_bonus    = scoring_config.geo_bonus  if  country_code == job.country_code
    hybrid       = w_semantic * cosine_sim + w_structured * (w_rate * rate_compat + w_rep * reputation) + geo_bonus
    """
    rate_usd:         Optional[float] = Field(None, description="Parsed hourly rate in USD")
    feedback_score:   float           = Field(...,  description="Feedback percentage normalised [0, 1]")
    jobs_done:        int             = Field(...,  description="Completed fixed-price jobs count")
    country_code:     str             = Field(...,  description="ISO-2 country code for geo bonus")
    reputation_score: float           = Field(...,  description="Pre-computed: 0.7*feedback + 0.3*min(jobs_done/50,1)")


class FreelancerProfileResponse(BaseModel):
    """
    Full freelancer profile — returned after processing backend-supplied data.

    Backend workflow
    ----------------
    1. POST freelancer data → receive + store this response.
    2. POST each job → receive + store job responses.
    3. At recommendation time (fully on backend):
       a. Filter jobs:  budget_label IN preferred_budget_range
                        AND client_country IN preferred_locations
       b. Rank filtered jobs:
            similarity   = cosine_similarity(this.embedding, job.embedding)
            rate_compat  = exp(-0.5 * ln(rate_usd / (job.budget_avg / weekly_hours))^2 / 0.64)
            reputation   = scoring_signals.reputation_score
            structured   = w_rate * rate_compat + w_rep * reputation
            geo          = geo_bonus if scoring_signals.country_code == job.country_code else 0
            hybrid_score = w_sem * similarity + w_struct * structured + geo  (clipped to 1.0)
       c. Sort by hybrid_score DESC, return top-N.
    """
    freelancer_id:          str
    embedding:              list[float]     = Field(..., description="Semantic vector for cosine similarity")
    preferred_budget_range: tuple[str, str] = Field(..., description="(min_label, max_label) — budget pre-filter")
    preferred_locations:    list[str]       = Field(..., description="Preferred client countries — geo pre-filter")
    scoring_signals:        FreelancerScoringSignals
    scoring_config:         dict[str, Any]  = Field(..., description="Exact weights + formulas to replicate hybrid scoring")


class JobEmbeddingResponse(BaseModel):
    """
    Job profile — returned after processing backend-supplied data.

    Filter fields (apply BEFORE similarity to narrow candidates):
      budget_label    → compare against freelancer.preferred_budget_range
      client_country  → compare against freelancer.preferred_locations

    Scoring fields (apply AFTER similarity to re-rank):
      budget_avg      → implied_rate = budget_avg / weekly_hours → rate_compat
      country_code    → compare to freelancer.country_code for geo bonus
    """
    job_id:         Optional[Any]
    job_title:      str
    embedding:      list[float]    = Field(..., description="Semantic vector for cosine similarity")
    budget_min:     float          = Field(..., description="Min posted budget (USD)")
    budget_max:     float          = Field(..., description="Max posted budget (USD)")
    budget_avg:     float          = Field(..., description="Avg budget (USD) — used in rate_compat formula")
    budget_label:   str            = Field(..., description="Budget tier: micro|small|medium|large|enterprise")
    client_country: str            = Field(..., description="Client country — filter against preferred_locations")
    client_state:   str            = Field(..., description="Client state/city")
    country_code:   str            = Field(..., description="ISO-2 code — compare to freelancer.country_code for geo bonus")
    client_rating:  float          = Field(..., description="Client average rating (0–5)")
    review_count:   int            = Field(..., description="Number of client reviews")
    tags:           str            = Field(..., description="Cleaned skill tags")
    scoring_config: dict[str, Any] = Field(..., description="Weights + formulas — same for all jobs")


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


# ── Freelancer profile — backend sends data, API processes + returns profile ──

@app.post(
    f"/{API_VERSION}/freelancers/profile",
    response_model=FreelancerProfileResponse,
)
async def embed_freelancer(req: FreelancerInput):
    """
    Process raw freelancer data sent by the backend and return a full profile.

    The backend sends the freelancer's raw fields.
    This endpoint:
      1. Cleans and enriches the text (job_title + skills + bio)
      2. Generates a semantic embedding via the ML model
      3. Predicts preferred_budget_range from the hourly rate
      4. Predicts preferred_locations from location + job demand data
      5. Computes all scoring signals (rate_usd, feedback_score, etc.)
      6. Returns everything the backend needs to store and rank later

    No stored data is used — everything is computed from the input.
    """
    _check_engine()

    preprocessor = engine._preprocessor
    embedder     = engine._embedder

    # 1. Clean and build enriched text from backend-supplied data
    row = preprocessor.process_freelancer_input(req.model_dump())

    # 2. Embed the enriched text
    emb = embedder.encode([row["enriched_text"]], desc="Freelancer embed")[0]

    # 3. Predict budget range from parsed rate
    from models.embedding_engine import _infer_budget_range, _infer_preferred_locations
    budget_range = _infer_budget_range(
        float(row["rate_usd"]) if row["rate_usd"] is not None else None
    )

    # 4. Predict preferred locations using job demand data
    locations = _infer_preferred_locations(
        row["location"], budget_range, engine.df_jobs, top_n=5
    )

    # 5. Compute scoring signals
    feedback   = float(row["feedback_score"] or 0)
    jobs_done  = int(row["jobs_done"] or 0)
    jobs_norm  = min(jobs_done / 50, 1.0)
    reputation = round(0.7 * feedback + 0.3 * jobs_norm, 4)
    rate_val   = float(row["rate_usd"]) if row["rate_usd"] is not None else None

    signals = FreelancerScoringSignals(
        rate_usd         = round(rate_val, 4) if rate_val is not None else None,
        feedback_score   = round(feedback, 4),
        jobs_done        = jobs_done,
        country_code     = str(row["country_code"] or ""),
        reputation_score = reputation,
    )

    return FreelancerProfileResponse(
        freelancer_id          = req.freelancer_id,
        embedding              = [round(float(x), 6) for x in emb],
        preferred_budget_range = budget_range,
        preferred_locations    = locations,
        scoring_signals        = signals,
        scoring_config         = SCORING_CONFIG,
    )


# ── Job embedding — backend sends data, API processes + returns job profile ───

@app.post(
    f"/{API_VERSION}/jobs/embed",
    response_model=JobEmbeddingResponse,
)
async def embed_job(req: JobInput):
    """
    Process raw job data sent by the backend and return a full job profile.

    The backend sends the job's raw fields.
    This endpoint:
      1. Cleans tags, parses budget fields, resolves country code
      2. Builds enriched text (title + tags + reputation tier + description)
      3. Generates a semantic embedding via the ML model
      4. Returns everything needed to filter and score against freelancers

    No stored data is used — everything is computed from the input.
    """
    _check_engine()

    preprocessor = engine._preprocessor
    embedder     = engine._embedder

    # 1. Clean and build enriched text from backend-supplied data
    row = preprocessor.process_job_input(req.model_dump())

    # 2. Embed
    emb = embedder.encode([row["enriched_text"]], desc="Job embed")[0]

    budget_avg = float(row["budget_avg"] or 0)
    budget_min = float(row["budget_min"] or 0)
    budget_max = float(row["budget_max"] or 0)

    return JobEmbeddingResponse(
        job_id         = req.job_id,
        job_title      = str(row["job_title"] or ""),
        embedding      = [round(float(x), 6) for x in emb],
        budget_min     = round(budget_min, 2),
        budget_max     = round(budget_max, 2),
        budget_avg     = round(budget_avg, 2),
        budget_label   = _budget_label(budget_avg),
        client_country = str(row["client_country"] or ""),
        client_state   = str(row["client_state"]   or ""),
        country_code   = str(row["country_code"]   or ""),
        client_rating  = round(float(row["client_rating"] or 0), 2),
        review_count   = int(row["review_count"] or 0),
        tags           = str(row["tags_cleaned"] or ""),
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
