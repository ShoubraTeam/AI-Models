"""
DataPreprocessor
================
Handles all data loading, cleaning, and enriched-text construction for
both the freelancers and jobs datasets.

Fixes identified in analysis
-----------------------------
1. skills_cleaned has trailing empty strings  →  stripped out
2. enriched_text duplicates the skill list    →  rebuilt from scratch
3. hour_rate / earnings stored as raw strings →  parsed to float
4. Client-reputation signals missing from job enriched_text  →  added
"""

import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Load, clean, and enrich raw CSV data for downstream embedding."""

    # ── Country normalisation map (extend as needed) ─────────────────────────
    COUNTRY_CODES: dict[str, str] = {
        "Pakistan": "PK", "India": "IN", "Germany": "DE",
        "United States": "US", "Bangladesh": "BD", "Philippines": "PH",
        "Egypt": "EG", "Nigeria": "NG", "Ukraine": "UA", "Brazil": "BR",
        "United Kingdom": "GB", "Canada": "CA", "Australia": "AU",
    }

    def __init__(self, freelancers_path: str | Path, jobs_path: str | Path) -> None:
        self.freelancers_path = Path(freelancers_path)
        self.jobs_path = Path(jobs_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_freelancer_input(self, data: dict) -> pd.Series:
        """
        Process a raw freelancer dict sent by the backend into a clean Series
        ready for embedding and scoring — no CSV needed.

        Expected input fields (all optional, missing = sensible default):
            job_title, skills, description, hour_rate, feedback_percent,
            fixed_jobs_done, location
        """
        skills_raw = data.get("skills") or data.get("skills_cleaned") or ""
        skills_cleaned = self._clean_skill_list(skills_raw)

        rate_usd      = self._parse_usd(data.get("hour_rate"))
        feedback_score = self._parse_percent(data.get("feedback_percent"))
        jobs_done     = self._parse_jobs_done(data.get("fixed_jobs_done"))
        location      = str(data.get("location") or "").strip()
        country_code  = self.COUNTRY_CODES.get(location, "")

        row = pd.Series({
            "job_title":      str(data.get("job_title") or ""),
            "skills_cleaned": skills_cleaned,
            "description":    str(data.get("description") or ""),
            "rate_usd":       rate_usd,
            "feedback_score": feedback_score,
            "jobs_done":      jobs_done,
            "location":       location,
            "country_code":   country_code,
        })
        row["enriched_text"] = self._build_freelancer_text(row)
        return row

    def process_job_input(self, data: dict) -> pd.Series:
        """
        Process a raw job dict sent by the backend into a clean Series
        ready for embedding and scoring — no CSV needed.

        Expected input fields (all optional, missing = sensible default):
            job_title, job_description, tags, client_country, client_state,
            client_average_rating, client_review_count,
            min_price, max_price, avg_price
        """
        import math as _math

        def _to_float(v):
            try:
                f = float(v)
                return 0.0 if _math.isnan(f) else f
            except (TypeError, ValueError):
                return 0.0

        def _to_int(v):
            try:
                f = float(v)
                return 0 if _math.isnan(f) else int(f)
            except (TypeError, ValueError):
                return 0

        tags_raw = data.get("tags") or data.get("tags_cleaned") or ""
        if isinstance(tags_raw, list):
            tags_cleaned = ", ".join(str(t).strip() for t in tags_raw if str(t).strip())
        else:
            tags_cleaned = self._clean_skill_list(tags_raw)

        client_country = str(data.get("client_country") or "").strip()
        country_code   = self.COUNTRY_CODES.get(client_country, "")

        budget_min = _to_float(data.get("min_price"))
        budget_max = _to_float(data.get("max_price"))
        budget_avg = _to_float(data.get("avg_price")) or (
            (budget_min + budget_max) / 2 if budget_min or budget_max else 0.0
        )

        row = pd.Series({
            "job_title":             str(data.get("job_title") or ""),
            "job_description":       str(data.get("job_description") or ""),
            "tags_cleaned":          tags_cleaned,
            "client_country":        client_country,
            "client_state":          str(data.get("client_state") or ""),
            "client_average_rating": _to_float(data.get("client_average_rating")),
            "client_review_count":   _to_int(data.get("client_review_count")),
            "client_rating":         _to_float(data.get("client_average_rating")),
            "review_count":          _to_int(data.get("client_review_count")),
            "budget_min":            budget_min,
            "budget_max":            budget_max,
            "budget_avg":            budget_avg,
            "country_code":          country_code,
        })
        row["enriched_text"] = self._build_job_text(row)
        return row

    def load_and_clean(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (df_freelancers, df_jobs) after full cleaning pipeline."""
        logger.info("Loading freelancers from %s", self.freelancers_path)
        df_f = pd.read_csv(self.freelancers_path)

        logger.info("Loading jobs from %s", self.jobs_path)
        df_j = pd.read_csv(self.jobs_path)

        df_f = self._clean_freelancers(df_f)
        df_j = self._clean_jobs(df_j)

        logger.info(
            "Loaded %d freelancers and %d jobs after cleaning",
            len(df_f), len(df_j),
        )
        return df_f, df_j

    # ── Freelancer cleaning ───────────────────────────────────────────────────

    def _clean_freelancers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Strip trailing empty strings from skills_cleaned
        df["skills_cleaned"] = df["skills_cleaned"].apply(self._clean_skill_list)

        # 2. Parse structured numeric fields
        df["rate_usd"]     = df["hour_rate"].apply(self._parse_usd)
        df["earnings_usd"] = df["earnings"].apply(self._parse_usd)
        df["feedback_score"] = df["feedback_percent"].apply(self._parse_percent)
        df["jobs_done"]    = df["fixed_jobs_done"].apply(self._parse_jobs_done)

        # 3. Rebuild enriched_text without duplication
        df["enriched_text"] = df.apply(self._build_freelancer_text, axis=1)

        # 4. Add country code for geo matching
        df["country_code"] = df["location"].apply(
            lambda x: self.COUNTRY_CODES.get(str(x).strip(), "")
        )

        return df

    def _build_freelancer_text(self, row: pd.Series) -> str:
        """Single, clean enriched text — no duplicate skill list."""
        parts = [
            str(row.get("job_title", "") or ""),
            str(row.get("skills_cleaned", "") or ""),
            str(row.get("description", "") or "")[:500],
        ]
        return " ".join(p.strip() for p in parts if p.strip()).lower()

    # ── Job cleaning ──────────────────────────────────────────────────────────

    def _clean_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Parse numeric fields
        df["client_rating"]  = pd.to_numeric(df["client_average_rating"], errors="coerce").fillna(0)
        df["review_count"]   = pd.to_numeric(df["client_review_count"],   errors="coerce").fillna(0)
        df["budget_avg"]     = pd.to_numeric(df["avg_price"], errors="coerce").fillna(0)
        df["budget_min"]     = pd.to_numeric(df["min_price"], errors="coerce").fillna(0)
        df["budget_max"]     = pd.to_numeric(df["max_price"], errors="coerce").fillna(0)

        # 2. Add client-reputation tier to enriched text (was missing)
        df["enriched_text"] = df.apply(self._build_job_text, axis=1)

        # 3. Country code for geo matching
        df["country_code"] = df["client_country"].apply(
            lambda x: self.COUNTRY_CODES.get(str(x).strip(), "")
        )

        return df

    def _build_job_text(self, row: pd.Series) -> str:
        """Include client-reputation signals in the embedding text."""
        import math as _math
        _r = row.get("client_average_rating")
        rating  = 0.0 if (_r is None or (isinstance(_r, float) and _math.isnan(_r))) else float(_r)
        _c = row.get("client_review_count")
        reviews = 0 if (_c is None or (isinstance(_c, float) and _math.isnan(_c))) else int(_c)
        country = str(row.get("client_country") or "").strip()

        if reviews > 50:
            reputation = "established client high reputation"
        elif reviews > 10:
            reputation = "verified client good track record"
        elif reviews > 0:
            reputation = "new client some reviews"
        else:
            reputation = "unrated client"

        parts = [
            str(row.get("job_title", "") or ""),
            str(row.get("tags_cleaned", "") or ""),
            reputation,
            f"client from {country}" if country else "",
            str(row.get("job_description", "") or "")[:500],
        ]
        return " ".join(p.strip() for p in parts if p.strip()).lower()

    # ── Field parsers ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean_skill_list(raw: object) -> str:
        """Remove empty tokens like ', , ,' from stringified lists."""
        if pd.isna(raw):
            return ""
        # Handle both stringified Python lists and plain CSV strings
        text = str(raw).strip("[]").replace("'", "")
        tokens = [t.strip() for t in text.split(",") if t.strip()]
        return ", ".join(tokens)

    @staticmethod
    def _parse_usd(raw: object) -> float | None:
        """Parse '$4.80', '$2K+ earned', '10M' → float dollars."""
        if pd.isna(raw):
            return None
        s = str(raw).replace(",", "").strip()
        m = re.search(r"([\d.]+)\s*([KkMm]?)", s)
        if not m:
            return None
        value = float(m.group(1))
        suffix = m.group(2).upper()
        if suffix == "K":
            value *= 1_000
        elif suffix == "M":
            value *= 1_000_000
        return value

    @staticmethod
    def _parse_percent(raw: object) -> float:
        """Parse '98%' → 0.98, handle NaN → 0.0."""
        if pd.isna(raw):
            return 0.0
        m = re.search(r"([\d.]+)", str(raw))
        return float(m.group(1)) / 100.0 if m else 0.0

    @staticmethod
    def _parse_jobs_done(raw: object) -> int:
        """Parse '9 fixed price jobs' → 9, NaN → 0."""
        if pd.isna(raw):
            return 0
        m = re.search(r"(\d+)", str(raw))
        return int(m.group(1)) if m else 0
