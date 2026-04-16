# Freelancer Job Recommender

A production-grade semantic + hybrid recommendation system that matches
freelancers to job postings (and vice versa) using dense embeddings,
approximate nearest-neighbour search, and structured signal re-ranking.

---

## What changed vs the original notebook

| Area | Before | After |
|---|---|---|
| **Matching** | Pure cosine similarity over all pairs | Hybrid score: 70% semantic + 30% structured (rate compatibility + reputation) |
| **Performance** | Brute-force O(n×m) per query | Qdrant HNSW index — O(log n) |
| **Evaluation** | Mean cosine similarity (poor proxy) | NDCG@K + Precision@K with Jaccard relevance |
| **Data quality** | Duplicate skills in enriched_text; empty skill tokens | Rebuilt enriched_text; empty tokens stripped; client reputation added to job text |
| **Parsed fields** | hour_rate/earnings as raw strings | Parsed to float for structured scoring |
| **Caching** | Saved to pickle but never loaded | Load-or-compute; incremental update for new records |
| **Direction** | Freelancer → Jobs only | Bidirectional: Job → Freelancers added |
| **Serving** | None | FastAPI REST API with `/recommend/jobs` and `/recommend/freelancers` |
| **Code structure** | Single notebook | Modular classes — one responsibility per file |

---

## Project structure

```
freelancer_recommender/
├── config/
│   └── settings.py          # All hyper-parameters and paths
├── data/
│   └── preprocessor.py      # DataPreprocessor — load, clean, enrich
├── models/
│   ├── embedding_engine.py  # EmbeddingEngine — encode + incremental cache
│   ├── vector_store.py      # VectorStore — Qdrant ANN wrapper
│   ├── scoring_engine.py    # ScoringEngine — hybrid re-ranking
│   └── recommendation_engine.py  # RecommendationEngine — main facade
├── evaluation/
│   └── evaluation_engine.py # EvaluationEngine — NDCG, Precision@K, plots
├── api/
│   └── server.py            # FastAPI serving layer
├── utils/
│   └── logging_setup.py     # Structured logging config
├── notebooks/
│   └── walkthrough.ipynb    # End-to-end demo notebook
├── data/                    # Put CSVs here
│   ├── freelancers_master_cleaned.csv
│   └── jobs_master_cleaned.csv
├── cache/                   # Auto-created — stores embeddings.pkl
├── logs/                    # Auto-created — stores recommender.log
├── main.py                  # CLI entry point
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your data

```bash
cp freelancers_master_cleaned.csv data/
cp jobs_master_cleaned.csv        data/
```

### 3. Run the full pipeline

```bash
python main.py
```

This will:
- Clean and enrich the data
- Encode embeddings (or load from cache on subsequent runs)
- Index vectors in Qdrant
- Print sample recommendations
- Run NDCG / Precision@K evaluation with plots

### 4. Force a full rebuild (if data changes)

```bash
python main.py --rebuild
```

### 5. Serve as a REST API

```bash
python main.py --serve
# API available at http://localhost:8000
# Docs at         http://localhost:8000/docs
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/v1/health` | Liveness check |
| `POST` | `/v1/recommend/jobs` | Top-N jobs for a freelancer |
| `POST` | `/v1/recommend/freelancers` | Top-N freelancers for a job |
| `GET`  | `/v1/freelancers/{id}` | Freelancer profile |
| `GET`  | `/v1/jobs/{index}` | Job record |

### Example request

```bash
curl -X POST http://localhost:8000/v1/recommend/jobs \
  -H "Content-Type: application/json" \
  -d '{"freelancer_index": 0, "top_n": 5}'
```

### Example response

```json
[
  {
    "rank": 1,
    "job_title": "Data Entry Specialist needed",
    "semantic_score": 0.8821,
    "structured_score": 0.7340,
    "hybrid_score": 0.8394,
    "geo_bonus_applied": false,
    "payload": { ... }
  }
]
```

---

## Configuration

All tuneable parameters are in `config/settings.py`:

```python
EMBEDDING_MODEL   = "BAAI/bge-base-en-v1.5"   # swap to bge-large for quality
WEIGHT_SEMANTIC   = 0.70    # semantic vs structured balance
WEIGHT_STRUCTURED = 0.30
GEO_BONUS         = 0.05    # flat bonus for country match
MIN_SCORE_THRESHOLD = 0.35  # filter low-confidence results
```

---

## Scoring formula

```
hybrid = 0.70 × semantic_cosine
       + 0.30 × (0.50 × rate_compatibility + 0.50 × reputation)
       + 0.05  [if freelancer country == client country]
```

**rate_compatibility** uses a log-Gaussian decay centred at a 1:1 ratio
between the freelancer's hourly rate and the job's implied rate
(`budget / 40 hrs`).

**reputation** blends `feedback_score` (70%) and normalised `jobs_done`
(30%, capped at 50 jobs = 1.0).

---

## Evaluation

Running `python main.py --eval` produces:

- **NDCG@10** — how well the ranked list matches Jaccard-relevance order
- **Precision@10** — fraction of top-10 results above the relevance median
- **MeanSim@10** — original metric, kept for backwards comparison
- Distribution histograms for NDCG and Precision

---

## Extending the system

- **Real relevance labels**: replace `EvaluationEngine._relevance_row()` with
  actual hire/click data when available.
- **Persistent Qdrant**: change `QDRANT_HOST = "localhost"` and run
  `docker run -p 6333:6333 qdrant/qdrant`.
- **Better model**: set `EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"` and
  `EMBEDDING_DIM = 1024`; run `python main.py --rebuild`.
- **LLM re-ranking**: pipe the top-20 ANN results into Gemini/GPT for a
  final natural-language re-rank pass.
