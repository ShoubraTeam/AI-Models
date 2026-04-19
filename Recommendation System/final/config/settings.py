"""
Central configuration for the Freelancer Job Recommender System.
All tuneable hyper-parameters and paths live here — import from this
module rather than hard-coding values anywhere else.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
CACHE_DIR  = BASE_DIR / "cache"
LOGS_DIR   = BASE_DIR / "logs"

FREELANCERS_CSV = DATA_DIR / "freelancers_master_cleaned.csv"
JOBS_CSV        = DATA_DIR / "jobs_master_cleaned.csv"
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings.pkl"

# ── Embedding model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "BAAI/bge-base-en-v1.5"   # swap to bge-large for higher quality
EMBEDDING_DIM     = 768
BATCH_SIZE        = 32
MAX_TEXT_LENGTH   = 512   # chars — trim noisy bios before encoding

# ── Vector store (Qdrant) ────────────────────────────────────────────────────
QDRANT_HOST            = ":memory:"   # use "localhost" for a persistent server
QDRANT_PORT            = 6333
QDRANT_JOBS_COLLECTION = "jobs"
QDRANT_FREELANCERS_COLLECTION = "freelancers"

# ── Matching / scoring ───────────────────────────────────────────────────────
DEFAULT_TOP_N      = 10
MIN_SCORE_THRESHOLD = 0.35

# Hybrid score weights  (must sum to 1.0)
WEIGHT_SEMANTIC    = 0.70
WEIGHT_STRUCTURED  = 0.30

# Structured sub-weights (must sum to 1.0)
WEIGHT_RATE_COMPAT = 0.50
WEIGHT_REPUTATION  = 0.50

# Geo bonus (added on top when freelancer country == client country)
GEO_BONUS          = 0.05

# Assumed weekly hours for rate → budget normalisation
WEEKLY_HOURS       = 40

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_TOP_N              = 10
EVAL_SAMPLE_FREELANCERS = 200   # use a subset for fast eval during dev
EVAL_SAMPLE_JOBS        = 500

# ── API ──────────────────────────────────────────────────────────────────────
API_HOST    = "0.0.0.0"
API_PORT    = 8000
API_VERSION = "v1"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = LOGS_DIR / "recommender.log"
