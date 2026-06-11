import inspect
from time import perf_counter
from pydantic import BaseModel
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# helpers & config
from helpers.config import ROUTE_MAIN_ROUTE
import helpers.functional as F

# messages & enums
from models.enums import ResponsesEnum, ErrorsEnum
from models.pydantic_schemas import AgentInferenceResult
from models.data_config import (
    PROFILE_SCORER_FEATURES_EXTRACTION,
    PROFILE_SCORER_FINAL_ANALYSIS
)

# controllers
from controllers import FeatureController
from controllers import AgentController


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
    image_path: str
    pre_extracted_features: dict


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
        return get_bad_request_profile_features(message = ResponsesEnum.GENERAL_ERROR_WRONG_FEATURE_ID.value)

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
            raise RuntimeError(f"Some profile features failed: {feature_errors}")

    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_profile_features(message = m)

    duration_s = perf_counter() - start_time
    try:
        result_to_log = AgentInferenceResult(
            user_input   = jsonable_encoder(profile_input_dict),
            agent_output = jsonable_encoder(agent_output),
            duration_s   = duration_s,
            task         = "profile_scorer_features_extraction"
        )
        feature_controller.log_result(result = result_to_log)
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
    request   : Request,
    data      : ProfileFinalAnalysisIP
) -> JSONResponse:

    start_time = perf_counter()
    task = PROFILE_SCORER_FINAL_ANALYSIS

    if not F.validate_feature_id(feature_id = feature_id):
        return get_bad_request_profile_final_analysis(message = ResponsesEnum.GENERAL_ERROR_WRONG_FEATURE_ID.value)

    feature_controller = FeatureController(feature_id = feature_id)
    agent_controller = AgentController(
        feature_id = feature_id,
        agents     = request.app.state.agents[feature_id],
        task       = task
    )

    profile_data = {
        "job_role": data.job_role,
        "hourly_rate": data.hourly_rate,
        "rating": data.rating,
        "total_completed_jobs": data.total_completed_jobs,
        "bio_text": data.bio_text,
        "declared_skills": data.declared_skills
    }
    img_path = data.image_path
    pre_extracted_features = data.pre_extracted_features

    try:
        preprocessed_data = await resolve_if_awaitable(
            agent_controller.preprocess_input(input = (profile_data, img_path, pre_extracted_features))
        )
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_PREPROCESSING_INPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_profile_final_analysis(message = m)

    try:
        agent_output = await resolve_if_awaitable(
            agent_controller.call_agent(input = (profile_data, preprocessed_data))
        )
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_profile_final_analysis(message = m)

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
        import os
        import mimetypes 
        from models.pydantic_schemas import ImageLog
        
        file_size_mb = 0.0
        content_type = "image/jpeg" 
        
        if os.path.exists(img_path):
            file_size_mb = round(os.path.getsize(img_path) / (1024 * 1024), 2)
            
            guessed_type, _ = mimetypes.guess_type(img_path)
            if guessed_type:
                content_type = guessed_type
            
        img_log_obj = ImageLog(
            filename     = os.path.basename(img_path),
            saved_path   = img_path,
            size_mbytes  = file_size_mb,
            content_type = content_type 
        )

        result_to_log = AgentInferenceResult(
            user_input   = f"Profile Data: {profile_data} | Image Path: {img_path}",
            task         = "profile_scorer_final_analysis",
            agent_output = final_report,
            duration_s   = duration_s,
            images       = (img_log_obj, img_log_obj)
        )
        feature_controller.log_result(result = result_to_log)
        
    except Exception as e:
        F.print_error(error = e, message = ErrorsEnum.DEBUG_ERROR_LOGGING_THE_RESULT.value)

    return get_good_request_profile_final_analysis(
        message = ResponsesEnum.PROFILE_SCORER_ANALYSIS_COMPLETED.value,
        report = final_report
    )