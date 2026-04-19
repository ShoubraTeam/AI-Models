"""
main.py
=======
Entry point for the Freelancer Job Recommender.

Usage
-----
# Build embeddings, run evaluation, print sample recommendations
python main.py

# Build and serve the REST API
python main.py --serve

# Force re-embed everything (ignore cache)
python main.py --rebuild
"""

import argparse
import logging

from utils.logging_setup import setup_logging
from models.recommendation_engine import RecommendationEngine
from evaluation.evaluation_engine import EvaluationEngine

setup_logging()
logger = logging.getLogger(__name__)


def run_demo(engine: RecommendationEngine) -> None:
    """Print sample recommendations for a few freelancers and jobs."""
    df_f = engine.df_freelancers

    print("\n" + "▓" * 60)
    print(" SAMPLE: Freelancer → Best Jobs")
    print("▓" * 60)
    for i in [0, 50, 200]:
        if i >= len(df_f):
            continue
        row = df_f.iloc[i]
        matches = engine.recommend_jobs(freelancer_index=i, top_n=5)
        engine.print_recommendations(
            matches,
            title=f"Jobs for {row['name']} | {row['job_title']}",
        )

    print("\n" + "▓" * 60)
    print(" SAMPLE: Job → Best Freelancers  (reverse matching)")
    print("▓" * 60)
    for job_idx in [0, 100]:
        matches = engine.recommend_freelancers(job_index=job_idx, top_n=5)
        job_title = engine.df_jobs.iloc[job_idx]["job_title"]
        engine.print_recommendations(
            matches,
            title=f"Freelancers for job: {job_title[:55]}",
        )


def run_evaluation(engine: RecommendationEngine) -> None:
    """Run the proper IR evaluation and show distribution plots."""
    evaluator = EvaluationEngine(
        df_freelancers=engine.df_freelancers,
        df_jobs=engine.df_jobs,
        freelancer_embeddings=engine._f_emb,
        job_embeddings=engine._j_emb,
    )
    summary = evaluator.evaluate()
    evaluator.plot_distribution()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freelancer Job Recommender")
    parser.add_argument("--serve",   action="store_true", help="Launch FastAPI server")
    parser.add_argument("--rebuild", action="store_true", help="Force re-embed all data")
    parser.add_argument("--eval",    action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    engine = RecommendationEngine()
    engine.build(recreate_index=args.rebuild)

    if args.serve:
        import uvicorn
        from api.server import app
        # Inject the already-built engine so the server doesn't rebuild
        import api.server as srv
        srv.engine = engine
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return

    if args.eval:
        run_evaluation(engine)
        return

    run_demo(engine)
    run_evaluation(engine)


if __name__ == "__main__":
    main()
