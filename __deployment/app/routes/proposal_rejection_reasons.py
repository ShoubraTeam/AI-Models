# ----------------------------------------------
# Serving Identity Recognition 
# ----------------------------------------------

# helpers
from helpers.config import ROUTE_MAIN_ROUTE
import helpers.functional as F
from time import perf_counter
import inspect

# messages
from models.enums            import ResponsesEnum, ErrorsEnum
from models.pydantic_schemas import AgentInferenceResult
from models.pydantic_schemas import JobToolResponse, JobKeyPointsSchema, ExtractedRequirementsSchema
from models.data_config      import (
    PROPOSAL_REJECTION_REASONS_JOB_FEATURES_EXTRACTION,
    PROPOSAL_REJECTION_REASONS_PROPOSAL_ANALYSIS
)

# controllers
from controllers import FeatureController
from controllers import AgentController

# fast api
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi import status
from fastapi.encoders import jsonable_encoder


# -------------------------- Helpers ---------------------------
async def resolve_if_awaitable(value):
    if inspect.isawaitable(value):
        return await value

    return value


def get_job_feature_errors(job_features: dict) -> dict[str, str]:
    return {
        feature_name: str(feature_result)
        for feature_name, feature_result in job_features.items()
        if isinstance(feature_result, Exception)
    }


def get_bad_request_job_features_extraction(message: str) -> JSONResponse:
    """Return a bad request error specific for job_features_extraction api"""
    return JSONResponse(
        status_code = status.HTTP_400_BAD_REQUEST,
        content = {
            "success"             : False,
            "message"             : message,
            "job_tools"            : None,
            "job_requirements"          : None,
            "job_key_points": None,
        }
    )

def get_good_request_job_analysis(message: str, job_features: dict) -> JSONResponse:
    """Return a good request specific for job_features_extraction api"""
    job_tools        = job_features['job_tools']
    job_requirements = job_features["job_requirements"]
    job_key_points   = job_features['job_key_points']
    
    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "success"         : True,
            "message"         : message,
            "job_tools"       : jsonable_encoder(job_tools),
            "job_requirements": jsonable_encoder(job_requirements),
            "job_key_points"  : jsonable_encoder(job_key_points),
        }
    )


def get_bad_request_proposal_analysis(message: str) -> JSONResponse:
    """Return a bad request error specific for proposal analysis api"""
    return JSONResponse(
        status_code = status.HTTP_400_BAD_REQUEST,
        content = {
            "success"             : False,
            "message"             : message,
            "proposal_report"     : None,
        }
    )

def get_good_request_proposal_analysis(message: str, report: str) -> JSONResponse:
    """Return a good request specific for proposal analysis api"""

    
    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "success"             : True,
            "message"             : message,
            "proposal_report"     : report,
        }
    )





# input schemas
from pydantic import BaseModel

class ExtractionJobFeaturesIP(BaseModel):
    job_description: str


class ProposalAnalysisIP(BaseModel):
    job_description: str
    proposal_text  : str
    job_features   : dict[str, JobToolResponse | JobKeyPointsSchema | ExtractedRequirementsSchema]

# -------------------------------- Routing ---------------------------------
proposal_rejection_reasons_router = APIRouter(
    prefix = ROUTE_MAIN_ROUTE 
)

@proposal_rejection_reasons_router.post("/{feature_id}/extract_job_features")
async def extract_job_features(
    feature_id: str,
    request   : Request,
    data      : ExtractionJobFeaturesIP
) -> JSONResponse:
    """
    This endpoint does the following:
        - Read the job
        - Extract features from that job
        - Return job results to be saved

    Returns:
        {
            "success"             : true if success else false.
            "message"             : message returned
            "job_tools"           : true if the same person, false if not, or None if success = False
            "job_requirements"    : similarity calculated between the two images
            "job_key_points"      : threshold determines Same vs Different
        }
    """
    start_time = perf_counter()

    # setup
    task = PROPOSAL_REJECTION_REASONS_JOB_FEATURES_EXTRACTION
    if not F.validate_feature_id(feature_id = feature_id):
        return get_bad_request_job_features_extraction(message = ResponsesEnum.GENERAL_ERROR_WRONG_FEATURE_ID.value)
    if not F.validate_proposal_rejection_reason_task(task = task):
        return get_bad_request_job_features_extraction(message = ErrorsEnum.PRR_ERROR_TASK.value)

    # controllers
    feature_controller = FeatureController(feature_id = feature_id)

    agent_controller_kwargs = {"task": task}
    agent_controller = AgentController(
        feature_id = feature_id,
        agents     = request.app.state.agents[feature_id],
        **agent_controller_kwargs
    )

    job_desc = data.job_description
    # calling the agent
    try:
        agent_output = await resolve_if_awaitable(
            agent_controller.call_agent(input = job_desc)
        )

        feature_errors = get_job_feature_errors(agent_output)
        if feature_errors:
            raise RuntimeError(f"Some job features failed: {feature_errors}")

    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_job_features_extraction(message = m)
    

    # log result
    end_time = perf_counter()
    duration_s = end_time - start_time

    try:
        result_to_log = AgentInferenceResult(
            user_input   = job_desc,
            agent_output = jsonable_encoder(agent_output),
            duration_s   = duration_s,
            task         = task
        )

        feature_controller.log_result(result = result_to_log)
    
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_LOGGING_THE_RESULT.value
        F.print_error(error = e, message = m)
        return get_bad_request_job_features_extraction(message = m)


    # return result
    return get_good_request_job_analysis(
        message = ResponsesEnum.PRR_JOB_FEATURES_EXTRACTED_CORRECTLY.value,
        job_features = agent_output
    )



# Proposal Analysis
@proposal_rejection_reasons_router.post("/{feature_id}/proposal_analysis")
async def recommend_tools(
    feature_id: str,
    data      : ProposalAnalysisIP,
    request   : Request
) -> JSONResponse:
    """
    This endpoint does the following:
        - Analyze the proposal quality given the job features
        - Construct a final report about the proposal

    Returns:
        {
            "success"        : True, if no error,
            "message"        : message returned,
            "proposal_report": if the job desc contain tools or not,
        }
    """
    start_time = perf_counter()


    # setup
    task = PROPOSAL_REJECTION_REASONS_PROPOSAL_ANALYSIS
    if not F.validate_feature_id(feature_id = feature_id):
        return get_bad_request_proposal_analysis(message = ResponsesEnum.GENERAL_ERROR_WRONG_FEATURE_ID.value)
    if not F.validate_proposal_rejection_reason_task(task = task):
        return get_bad_request_proposal_analysis(message = ErrorsEnum.PRR_ERROR_TASK.value)
    
    # controllers
    feature_controller = FeatureController(feature_id = feature_id)

    agent_controller_kwargs = {
        "task": task,
    }

    agent_controller   = AgentController(
        feature_id      = feature_id,
        agents          = request.app.state.agents[feature_id],
        **agent_controller_kwargs
    )


    # read data
    job_desc     = data.job_description
    proposal     = data.proposal_text
    job_features = data.job_features


    # preprocessing
    try:
        preprocessed = await resolve_if_awaitable(
            agent_controller.preprocess_input(input = (job_desc, proposal, job_features))
        )

    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_PREPROCESSING_INPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_proposal_analysis(message = m)


    # calling the agent
    try:
        agent_output = await resolve_if_awaitable(
            agent_controller.call_agent(input = (job_desc, proposal, preprocessed))
        )
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_proposal_analysis(message = m)
    

    # post-processing
    try:
        agent_output = await resolve_if_awaitable(
            agent_controller.postprocess_agent_output(agent_output = agent_output)
        )
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_POSTPROCESSING_OUTPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_proposal_analysis(message = m)
    

    # log result
    end_time = perf_counter()
    duration_s = end_time - start_time

    try:
        result_to_log = AgentInferenceResult(
            user_input   = (f"Job Description: {job_desc}", f"Proposal: {proposal}", f"Job Features: {jsonable_encoder(job_features)}"),
            task         = task,
            agent_output = agent_output,
            duration_s   = duration_s,
        )

        feature_controller.log_result(result = result_to_log)
    
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_LOGGING_THE_RESULT.value
        F.print_error(error = e, message = m)
        return get_bad_request_proposal_analysis(message = m)


    return get_good_request_proposal_analysis(
        message = ResponsesEnum.PRR_PROPOSAL_ANALYSIS_COMPLETED.value,
        report  = agent_output
    )
