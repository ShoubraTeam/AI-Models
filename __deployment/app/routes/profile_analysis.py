import inspect
from time import perf_counter
from pydantic import BaseModel
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import UploadFile, File

# helpers & config
from helpers.settings import ROUTE_MAIN_ROUTE
import helpers.functional as F

# messages & enums
from models.enums import ResponsesEnum, ErrorsEnum
from models.schemas import AgentResultsToSave, AgentInput, AgentOutput
from models.config.system_tasks import (
    PROFILE_SCORER_FEATURES_EXTRACTION,
    PROFILE_SCORER_FINAL_ANALYSIS
)

# controllers
from controllers.feature_controller import FeatureController
from controllers.agents_controller import AgentController

import os
import mimetypes 
from models.schemas import ImageLog

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
async def resolve_if_awaitable(value):
    if inspect.isawaitable(value):
        return await value
    return value


def get_bad_request_profile_features(message: str) -> JSONResponse:
    return JSONResponse(
        status_code = status.HTTP_400_BAD_REQUEST,
        content = {
            "success": False,
            "message": message,
            "numerical_res": None,
            "bio_res": None,
            "skills_res": None,
        }
    )


def get_good_request_profile_features(message: str, features: dict) -> JSONResponse:
    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "success": True,
            "message": message,
            "numerical_res": jsonable_encoder(features.get("numerical_res")),
            "bio_res": jsonable_encoder(features.get("bio_res")),
            "skills_res": jsonable_encoder(features.get("skills_res")),
        }
    )


def get_bad_request_profile_final_analysis(message: str) -> JSONResponse:
    return JSONResponse(
        status_code = status.HTTP_400_BAD_REQUEST,
        content = {
            "success": False,
            "message": message,
            "profile_report": None,
        }
    )


def get_good_request_profile_final_analysis(message: str, report: str) -> JSONResponse:
    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "success": True,
            "message": message,
            "profile_report": report,
        }
    )


# --------------------------------------------------------------------------
# Input Schemas (Pydantic Models)
# --------------------------------------------------------------------------
class ProfileFeaturesExtractionIP(BaseModel):
    job_role: str
    hourly_rate: float
    rating: float
    total_completed_jobs: int
    bio_text: str
    declared_skills: list[str]


class ProfileFinalAnalysisIP(BaseModel):
    job_role: str
    hourly_rate: float
    rating: float
    total_completed_jobs: int
    bio_text: str
    declared_skills: list[str]
    pre_extracted_features: dict




# --------------------------------------------------------------------------
# Saving Results
# --------------------------------------------------------------------------
def get_result_to_save(
    task                    : str,
    duration                : float,
    job_role                : str       | None = None,
    hourly_rate             : float     | None = None,
    rating                  : float     | None = None,
    total_completed_jobs    : int       | None = None,
    bio_text                : str       | None = None,
    declared_skills         : list[str] | None = None,
    user_feedback           : str       | None = None,
    numerical_res           : dict      | None = None,
    bio_res                 : dict      | None = None,
    skills_res              : dict      | None = None,
    profile_img             : ImageLog  | None = None,
    pre_extracted_features  : dict      | None = None,
    report                  : str       | None = None
) -> AgentResultsToSave:
    
    if task == PROFILE_SCORER_FEATURES_EXTRACTION:
        agent_input = AgentInput(
            input_id = f"profile_info",
            value    = {
                "job_role"            : job_role,
                "hourly_rate"         : hourly_rate,
                "rating"              : rating,
                "total_completed_jobs": total_completed_jobs,
                "bio_text"            : bio_text,
                "declared_skills"     : declared_skills,
            }
        )

        agent_output = AgentOutput(
            output_id = "profile_features",
            value = {
                "numerical_res": jsonable_encoder(numerical_res),
                "bio_res"      : jsonable_encoder(bio_res),
                "skills_res"   : jsonable_encoder(skills_res)
            }
        )

    elif task == PROFILE_SCORER_FINAL_ANALYSIS:
        agent_input = AgentInput(
            input_id = "whole_profile_data",
            value    = {
                "job_role"              : job_role,
                "profile_photo"         : profile_img.model_dump_json(indent = 2),
                "hourly_rate"           : hourly_rate,
                "rating"                : rating,
                "total_completed_jobs"  : total_completed_jobs,
                "bio_text"              : bio_text,
                "declared_skills"       : declared_skills,
                "pre_extracted_features": jsonable_encoder(pre_extracted_features)
            }
        )

        agent_output = AgentOutput(
            output_id = "fina_report",
            value     = report
        )



    return AgentResultsToSave(
        task = task,
        agent_input  = agent_input,
        agent_output = agent_output,
        duration_s = duration,
        user_feedback = user_feedback
    )

# --------------------------------------------------------------------------
# Routing Definitions
# --------------------------------------------------------------------------
profile_analysis_router = APIRouter(
    prefix = ROUTE_MAIN_ROUTE 
)


# --- Endpoint 1: Profile Features Extraction ---
@profile_analysis_router.post("/{feature_id}/profile_features_extraction")
async def profile_features_extraction(
    feature_id: str,
    request   : Request,
    data      : ProfileFeaturesExtractionIP
) -> JSONResponse:
    
    start_time = perf_counter()
    task = PROFILE_SCORER_FEATURES_EXTRACTION
    
    if not F.validate_feature_id(feature_id = feature_id):
        return get_bad_request_profile_features(message = ErrorsEnum.GENERAL_Invalid_FEATURE_ID.value)
    if not F.validate_profile_analysis_task(task = task):
        return get_bad_request_profile_features(message = ErrorsEnum.PS_INVALID_TASK.value)

    feature_controller = FeatureController(feature_id = feature_id)
    agent_controller = AgentController(
        feature_id = feature_id,
        agents     = request.app.state.agents[feature_id],
        task       = task
    )

    profile_input_dict = data.model_dump()

    try:
        agent_output = await resolve_if_awaitable(
            agent_controller.call_agent(input = profile_input_dict)
        )
        
        feature_errors = {k: str(v) for k, v in agent_output.items() if isinstance(v, Exception)}
        if feature_errors:
            raise RuntimeError(f"{ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value} - Some profile features failed: {feature_errors}")

    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_profile_features(message = m)

    duration_s = perf_counter() - start_time
    try:
        result_to_save = get_result_to_save(
            task = task,
            duration = duration_s,
            bio_res = agent_output["bio_res"],
            numerical_res = agent_output["numerical_res"],
            skills_res = agent_output["skills_res"],
            **profile_input_dict
        )
        feature_controller.log_result(result = result_to_save)
    except Exception as e:
        F.print_error(error = e, message = ErrorsEnum.DEBUG_ERROR_LOGGING_THE_RESULT.value)

    return get_good_request_profile_features(
        message = ResponsesEnum.PROFILE_SCORER_FEATURES_EXTRACTED_CORRECTLY.value,
        features = agent_output
    )


# --- Endpoint 2: Profile Final Analysis (SuperAgent) ---
@profile_analysis_router.post("/{feature_id}/profile_final_analysis")
async def profile_final_analysis(
    feature_id: str,
    request     : Request,
    data        : ProfileFinalAnalysisIP,
    profile_img : UploadFile = File(...)
) -> JSONResponse:

    start_time = perf_counter()
    task = PROFILE_SCORER_FINAL_ANALYSIS

    if not F.validate_feature_id(feature_id = feature_id):
        return get_bad_request_profile_features(message = ErrorsEnum.GENERAL_Invalid_FEATURE_ID.value)
    if not F.validate_profile_analysis_task(task = task):
        return get_bad_request_profile_features(message = ErrorsEnum.PS_INVALID_TASK.value)

    feature_controller = FeatureController(feature_id = feature_id)
    agent_controller = AgentController(
        feature_id = feature_id,
        agents     = request.app.state.agents[feature_id],
        task       = task
    )

    # read data
    profile_data = {
        "job_role": data.job_role,
        "hourly_rate": data.hourly_rate,
        "rating": data.rating,
        "total_completed_jobs": data.total_completed_jobs,
        "bio_text": data.bio_text,
        "declared_skills": data.declared_skills
    }

    profile_img_log = ImageLog(
        filename     = str(profile_img.filename),
        content_type = str(profile_img.content_type)
    )
    
    pre_extracted_features = data.pre_extracted_features


    try:
        profile_img = await profile_img.read()  ### !!! I edited it [no longer an img path, but real uploaded img]
    except Exception as e:
        F.print_error(error = e, message = ErrorsEnum.DEBUG_ERROR_LOADING_DATA.value)
        return get_bad_request_profile_final_analysis(message = ResponsesEnum.PROFILE_SCORER_IMAGE_LOADING_ERROR.value)

    profile_img_log.size_mbytes = len(profile_img) / (1024 * 1024)


    # pre-process
    try:
        preprocessed_data = await resolve_if_awaitable(
            agent_controller.preprocess_input(input = (profile_data, profile_img, pre_extracted_features))
        )

    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_PREPROCESSING_INPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_profile_final_analysis(message = m)


    # call
    try:
        agent_output = await resolve_if_awaitable(
            agent_controller.call_agent(input = (profile_data, preprocessed_data))
        )
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_profile_final_analysis(message = m)


    # post-process
    try:
        final_report = await resolve_if_awaitable(
            agent_controller.postprocess_agent_output(agent_output = agent_output)
        )
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_POSTPROCESSING_OUTPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_profile_final_analysis(message = m)

    duration_s = perf_counter() - start_time
    
    try:
        result_to_save = get_result_to_save(
            task = task,
            duration                = duration_s,
            report                  = final_report,
            profile_img             = profile_img_log,
            pre_extracted_features  = pre_extracted_features,
            **profile_data
        )

        feature_controller.log_result(result = result_to_save)
        
    except Exception as e:
        F.print_error(error = e, message = ErrorsEnum.DEBUG_ERROR_LOGGING_THE_RESULT.value)

    return get_good_request_profile_final_analysis(
        message = ResponsesEnum.PROFILE_SCORER_ANALYSIS_COMPLETED.value,
        report = final_report
    )