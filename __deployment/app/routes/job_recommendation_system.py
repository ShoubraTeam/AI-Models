# ----------------------------------------------
# Job Recommendation System
# ----------------------------------------------

# helpers
from helpers.settings import ROUTE_MAIN_ROUTE
import helpers.functional as F
from time import perf_counter

# messages
from models.enums   import ErrorsEnum
from models.schemas import AgentResultsToSave, AgentInput, AgentOutput
from models.schemas import FreelancerEmbedIP, JobEmbedIP
from models.config.system_tasks import (
    RS_FREELANCER_EMBEDDING,
    RS_JOB_EMBEDDING,
)

# controllers
from controllers.feature_controller import FeatureController
from controllers.agents_controller import AgentController


# fast api
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi import status



# -------------------------- Helper Functions ---------------------------
# freelancer embedding
def get_bad_request_freelancer_embedding(message: str) -> JSONResponse:
    """Return a bad request error specific for freelancer embedding api"""
    return JSONResponse(
        status_code = status.HTTP_400_BAD_REQUEST,
        content = {
            "success"      : False,
            "message"      : message,
            "enriched_text": None,
            "embedding"    : None,
            "embedding_dim": None,
        }
    )

def get_good_request_freelancer_embedding(message: str, enriched_text: str, embedding: list[float]) -> JSONResponse:
    """Return a good request specific for freelancer embedding"""

    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "success"      : True,
            "message"      : message,
            "enriched_text": enriched_text,
            "embedding"    : embedding,
            "embedding_dim": len(embedding),
        }
    )


# job embedding
def get_bad_request_job_embedding(message: str) -> JSONResponse:
    """Return a bad request error specific for job embedding api"""
    return JSONResponse(
        status_code = status.HTTP_400_BAD_REQUEST,
        content = {
            "success"      : False,
            "message"      : message,
            "enriched_text": None,
            "embedding"    : None,
            "embedding_dim": None,
        }
    )

def get_good_request_job_embedding(message: str, enriched_text: str, embedding: list[float]) -> JSONResponse:
    """Return a good request specific for job embedding"""

    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "success"      : True,
            "message"      : message,
            "enriched_text": enriched_text,
            "embedding"    : embedding,
            "embedding_dim": len(embedding),
        }
    )


def get_result_to_save(
    task         : str,
    duration     : float,
    bio          : str | None = None,
    description  : str | None = None,
    skills       : list[str] | None = None,
    job_title    : str | None = None,
    enriched_text: str | None = None,
    embedding    : list[float] | None = None,
    user_feedback: None | str = None
) -> AgentResultsToSave:
    
    if task == RS_FREELANCER_EMBEDDING:
        agent_input = AgentInput(
            input_id = "freelancer_embedding_input___bio___skills___job_title",
            value    = {
                "bio"      : bio,
                "skills"   : skills,
                "job_title": job_title,
            }
        )

        agent_output = AgentOutput(
            output_id = "freelancer_embedding",
            value     = {
                "enriched_text": enriched_text,
                "embedding"    : embedding,
                "embedding_dim": len(embedding) if embedding is not None else None,
            }
        )
    
    elif task == RS_JOB_EMBEDDING:
        agent_input = AgentInput(
            input_id = "job_embedding_input___job_description___skills___job_title",
            value    = {
                "description": description,
                "skills"     : skills,
                "job_title"  : job_title,
            }
        )

        agent_output = AgentOutput(
            output_id = "job_embedding",
            value     = {
                "enriched_text": enriched_text,
                "embedding"    : embedding,
                "embedding_dim": len(embedding) if embedding is not None else None,
            }
        )


    return AgentResultsToSave(
        task = task,
        agent_input  = agent_input,
        agent_output = agent_output,
        duration_s = duration,
        user_feedback = user_feedback
    )

# -------------------------------- Routing ---------------------------------
job_recommendation_system_router = APIRouter(prefix = ROUTE_MAIN_ROUTE )


# Freelancer Embedding
@job_recommendation_system_router.post("/{feature_id}/freelancer_embedding")
async def freelancer_embedding(
    feature_id: str,
    data      : FreelancerEmbedIP,
    request   : Request
) -> dict[str, bool]:
    """
    This endpoint does the following:
        - validate the given freelancer data
        - preprocess freelancer data
        - generate an embedding for recommendation

    Returns:
        {
            "success"      : True, if no error,
            "message"      : message returned,
            "enriched_text": enriched text used for embedding,
            "embedding"    : embedding vector,
            "embedding_dim": embedding vector length,
        }
    """
    start_time = perf_counter()

    # setup
    task = RS_FREELANCER_EMBEDDING
    if not F.validate_feature_id(feature_id = feature_id):
        return get_bad_request_freelancer_embedding(message = ErrorsEnum.GENERAL_Invalid_FEATURE_ID.value)
    if not F.validate_recommendation_system_task(task = task):
        return get_bad_request_freelancer_embedding(message = ErrorsEnum.RS_INVALID_TASK.value)
    
    # controllers
    feature_controller = FeatureController(feature_id = feature_id)

    agent_controller_kwargs = {"task": task}
    agent_controller = AgentController(
        feature_id = feature_id,
        agents     = request.app.state.agents[feature_id],
        **agent_controller_kwargs
    )    


    # read data
    bio       = data.bio
    skills    = data.skills
    job_title = data.job_title


    # preprocessing
    try:
        preprocessed = agent_controller.preprocess_input(input = (bio, skills, job_title))

    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_PREPROCESSING_INPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_freelancer_embedding(message = m)


    # calling the agent
    try:
        agent_output = agent_controller.call_agent(input = preprocessed)
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_freelancer_embedding(message = m)
    

    # post-processing
    try:
        agent_output = agent_controller.postprocess_agent_output(agent_output = agent_output)
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_POSTPROCESSING_OUTPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_freelancer_embedding(message = m)
    

    enriched_text = agent_output["enriched_text"]
    embedding     = agent_output["embedding"]


    # log result
    end_time = perf_counter()
    duration_s = end_time - start_time

    try:
        result_to_save = get_result_to_save(
            task          = task,
            duration      = duration_s,
            bio           = bio,
            skills        = skills,
            job_title     = job_title,
            enriched_text = enriched_text,
            embedding     = embedding
        )

        feature_controller.log_result(result = result_to_save)
    
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_LOGGING_THE_RESULT.value
        F.print_error(error = e, message = m)
        return get_bad_request_freelancer_embedding(message = m)


    return get_good_request_freelancer_embedding(
        message       = f"Freelancer embedding generated successfully in {duration_s:.4f}s.",
        enriched_text = enriched_text,
        embedding     = embedding
    )



# Job Embedding
@job_recommendation_system_router.post("/{feature_id}/job_embedding")
async def job_embedding(
    feature_id: str,
    data      : JobEmbedIP,
    request   : Request
) -> dict[str, bool]:
    """
    This endpoint does the following:
        - validate the given job data
        - preprocess job data
        - generate an embedding for recommendation

    Returns:
        {
            "success"      : True, if no error,
            "message"      : message returned,
            "enriched_text": enriched text used for embedding,
            "embedding"    : embedding vector,
            "embedding_dim": embedding vector length,
        }
    """
    start_time = perf_counter()

    # setup
    task = RS_JOB_EMBEDDING
    if not F.validate_feature_id(feature_id = feature_id):
        return get_bad_request_job_embedding(message = ErrorsEnum.GENERAL_Invalid_FEATURE_ID.value)
    if not F.validate_recommendation_system_task(task = task):
        return get_bad_request_job_embedding(message = ErrorsEnum.RS_INVALID_TASK.value)
    
    # controllers
    feature_controller = FeatureController(feature_id = feature_id)

    agent_controller_kwargs = {"task": task}
    agent_controller = AgentController(
        feature_id = feature_id,
        agents     = request.app.state.agents[feature_id],
        **agent_controller_kwargs
    )    


    # read data
    description = data.description
    skills      = data.skills
    job_title   = data.job_title


    # preprocessing
    try:
        preprocessed = agent_controller.preprocess_input(input = (description, skills, job_title))

    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_PREPROCESSING_INPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_job_embedding(message = m)


    # calling the agent
    try:
        agent_output = agent_controller.call_agent(input = preprocessed)
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_job_embedding(message = m)
    

    # post-processing
    try:
        agent_output = agent_controller.postprocess_agent_output(agent_output = agent_output)
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_POSTPROCESSING_OUTPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_job_embedding(message = m)
    

    enriched_text = agent_output["enriched_text"]
    embedding     = agent_output["embedding"]


    # log result
    end_time = perf_counter()
    duration_s = end_time - start_time

    try:
        result_to_save = get_result_to_save(
            task          = task,
            duration      = duration_s,
            description   = description,
            skills        = skills,
            job_title     = job_title,
            enriched_text = enriched_text,
            embedding     = embedding
        )

        feature_controller.log_result(result = result_to_save)
    
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_LOGGING_THE_RESULT.value
        F.print_error(error = e, message = m)
        return get_bad_request_job_embedding(message = m)


    return get_good_request_job_embedding(
        message       = f"Job embedding generated successfully in {duration_s:.4f}s.",
        enriched_text = enriched_text,
        embedding     = embedding
    )
