# ---------------------------------------------------------------
# Job Recommendation System Routes
# ---------------------------------------------------------------

from time import perf_counter

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from helpers.config import ROUTE_MAIN_ROUTE
import helpers.functional as F

from models.pydantic_schemas import FreelancerEmbedIP, JobEmbedIP
from models.data_config import (
    FEATURE_JOB_RECOMMENDATION_SYSTEM,
    RS_FREELANCER_EMBEDDING,
    RS_JOB_EMBEDDING,
)
from controllers import AgentController


job_recommendation_system_router = APIRouter(
    prefix=ROUTE_MAIN_ROUTE,
    tags=["Job Recommendation System"],
)


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success"      : False,
            "message"      : message,
            "enriched_text": None,
            "embedding"    : None,
            "embedding_dim": None,
        },
    )


def _internal_error(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success"      : False,
            "message"      : message,
            "enriched_text": None,
            "embedding"    : None,
            "embedding_dim": None,
        },
    )


def _ok(message: str, enriched_text: str, embedding: list[float]) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success"      : True,
            "message"      : message,
            "enriched_text": enriched_text,
            "embedding"    : embedding,
            "embedding_dim": len(embedding),
        },
    )


@job_recommendation_system_router.post(
    "/job-recommendation-system/freelancer-embedding",
    summary="Embed a freelancer profile",
)
async def freelancer_embedding(body: FreelancerEmbedIP, request: Request):
    t0 = perf_counter()

    try:
        agents = request.app.state.agents[FEATURE_JOB_RECOMMENDATION_SYSTEM]
        controller = AgentController(
            feature_id=FEATURE_JOB_RECOMMENDATION_SYSTEM,
            agents=agents,
            task=RS_FREELANCER_EMBEDDING,
        )
    except Exception as e:
        F.print_error(e, "Failed to initialise RS AgentController")
        return _internal_error("Recommendation System agent is not available.")

    try:
        raw_input    = (body.bio, body.skills, body.job_title)
        preprocessed = controller.preprocess_input(raw_input)
        agent_output = controller.call_agent(preprocessed)
        result       = controller.postprocess_agent_output(agent_output)
    except Exception as e:
        F.print_error(e, "Error during freelancer embedding pipeline")
        return _internal_error(f"Embedding pipeline failed: {e}")

    duration = round(perf_counter() - t0, 4)
    return _ok(
        message=f"Freelancer embedding generated successfully in {duration}s.",
        enriched_text=result["enriched_text"],
        embedding=result["embedding"],
    )


@job_recommendation_system_router.post(
    "/job-recommendation-system/job-embedding",
    summary="Embed a job posting",
)
async def job_embedding(body: JobEmbedIP, request: Request):
    t0 = perf_counter()

    try:
        agents = request.app.state.agents[FEATURE_JOB_RECOMMENDATION_SYSTEM]
        controller = AgentController(
            feature_id=FEATURE_JOB_RECOMMENDATION_SYSTEM,
            agents=agents,
            task=RS_JOB_EMBEDDING,
        )
    except Exception as e:
        F.print_error(e, "Failed to initialise RS AgentController")
        return _internal_error("Recommendation System agent is not available.")

    try:
        raw_input    = (body.description, body.skills, body.job_title)
        preprocessed = controller.preprocess_input(raw_input)
        agent_output = controller.call_agent(preprocessed)
        result       = controller.postprocess_agent_output(agent_output)
    except Exception as e:
        F.print_error(e, "Error during job embedding pipeline")
        return _internal_error(f"Embedding pipeline failed: {e}")

    duration = round(perf_counter() - t0, 4)
    return _ok(
        message=f"Job embedding generated successfully in {duration}s.",
        enriched_text=result["enriched_text"],
        embedding=result["embedding"],
    )