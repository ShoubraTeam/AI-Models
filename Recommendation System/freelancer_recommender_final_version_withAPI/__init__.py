"""
Freelancer Job Recommender
==========================
A production-grade semantic + hybrid job recommendation system.

Quick start
-----------
    from models import RecommendationEngine

    engine = RecommendationEngine()
    engine.build()

    jobs = engine.recommend_jobs(freelancer_index=0, top_n=10)
    for match in jobs:
        print(match.summary())
"""
