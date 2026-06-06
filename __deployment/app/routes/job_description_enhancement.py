# ----------------------------------------------
# Job Desc Enhancement
# ----------------------------------------------

# helpers
from helpers.config import ROUTE_MAIN_ROUTE
import helpers.functional as F
from time import perf_counter

# messages
from models.enums            import ResponsesEnum, ErrorsEnum
from models.pydantic_schemas import AgentInferenceResult
from models.pydantic_schemas import JobEnhancementIP, ToolsDetectionIP, ToolsRecommendationIP


# controllers
from controllers import FeatureController
from controllers import AgentController
from controllers import WeaviateController

# fast api
from fastapi import APIRouter, Request
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import status


# -------------------------- Helper Functions ---------------------------
def get_bad_request_tools_detection(message: str) -> JSONResponse:
    """Return a bad request error specific for job_desc_enhancement api"""
    return JSONResponse(
        status_code = status.HTTP_400_BAD_REQUEST,
        content = {
            "success"        : False,
            "message"        : message,
            "has_tools"      : None,
            "suggested_tools": None,
        }
    )

def get_good_request_tools_detection(message: str, has_tools: bool) -> JSONResponse:
    """Return a good request specific for tools_detection"""

    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "success"        : True,
            "message"        : message,
            "has_tools"      : has_tools,
        }
    )

def get_bad_request_tools_recommendation(message: str) -> JSONResponse:
    """Return a bad request error specific for job_desc_enhancement api"""
    return JSONResponse(
        status_code = status.HTTP_400_BAD_REQUEST,
        content = {
            "success": False,
            "message": message,
            "tools"  : None,
        }
    )

def get_good_request_tools_recommendation(message: str, tools: list[str]) -> JSONResponse:
    """Return a good request specific for tools_detection"""

    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "success": True,
            "message": message,
            "tools"  : tools,
        }
    )


def get_bad_request_job_desc_enhancement(message: str) -> JSONResponse:
    """Return a bad request error specific for job_desc_enhancement api"""
    return JSONResponse(
        status_code = status.HTTP_400_BAD_REQUEST,
        content = {
            "success"                  : False,
            "message"                  : message,
            "enhanced_job_description" : None,
        }
    )

def get_good_request_job_desc_enhancement(message: str, enhanced_job_description: str) -> JSONResponse:
    """Return a good request specific for tools_detection"""

    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "success"                  : True,
            "message"                  : message,
            "enhanced_job_description" : enhanced_job_description,
        }
    )

# -------------------------------- Routing ---------------------------------
job_description_enhancement_router = APIRouter(
    prefix = ROUTE_MAIN_ROUTE 
)


# Tools Detection
@job_description_enhancement_router.post("/{feature_id}/tools_detection")
async def detect_tools(
    feature_id: str,
    data      : ToolsDetectionIP,
    request   : Request
) -> dict[str, bool]:
    """
    This endpoint does the following:
        - validate the given job data
        - detecting tools in the desc
        - If there are tools -> return them
        - If there are not   -> return tools relevant to the job

    Returns:
        {
            "success"        : True, if no error,
            "message"        : message returned,
            "has_tools"      : if the job desc contain tools or not,
        }
    """
    task = "tools_detection"

    # setup
    start_time = perf_counter()

    if not F.validate_feature_id(feature_id = feature_id):
        return get_bad_request_tools_detection(message = ResponsesEnum.GENERAL_ERROR_WRONG_FEATURE_ID.value)
    
    # controllers
    feature_controller = FeatureController(feature_id = feature_id)

    agent_controller_kwargs = {
        "task": task
    }
    agent_controller   = AgentController(
        feature_id = feature_id,
        agents     = request.app.state.agents[feature_id],
        client     = request.app.state.groq_client,
        **agent_controller_kwargs
    )    


    # read data
    job_desc  = data.job_description


    # calling the agent
    try:
        agent_output = agent_controller.call_agent(input = job_desc)
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_tools_detection(message = m)
    

    # log result
    end_time = perf_counter()
    duration_s = end_time - start_time

    try:
        result_to_log = AgentInferenceResult(
            user_input   = job_desc,
            task         = task,
            agent_output = agent_output,
            duration_s   = duration_s,
        )

        feature_controller.log_result(result = result_to_log)
    
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_LOGGING_THE_RESULT.value
        F.print_error(error = e, message = m)
        return get_bad_request_tools_detection(message = m)


    return get_good_request_tools_detection(
        message   = ResponsesEnum.JD_ENH_SUCCESS_TOOLS_DETECTION.value,
        has_tools = agent_output
    )


# Tools Recommendation
@job_description_enhancement_router.post("/{feature_id}/tools_recommendation")
async def recommend_tools(
    feature_id: str,
    data      : ToolsRecommendationIP,
    request   : Request
) -> dict[str, bool]:
    """
    This endpoint does the following:
        - validate the given job data
        - Retreiving jobs relevant to the given job desc
        - Recommend common tools among those jobs

    Returns:
        {
            "success": True, if no error,
            "message": message returned,
            "tools"  : if the job desc contain tools or not,
        }
    """
    task = "tools_recommendation"

    # setup
    start_time = perf_counter()
    print(100*'=')
    print("here")
    print(100*'=')

    if not F.validate_feature_id(feature_id = feature_id):
        return get_bad_request_tools_recommendation(message = ResponsesEnum.GENERAL_ERROR_WRONG_FEATURE_ID.value)
    
    # controllers
    feature_controller = FeatureController(feature_id = feature_id)

    weaviate_controller = WeaviateController(
        agents = request.app.state.agents[feature_id],
        client = request.app.state.weaviate_client
    ) 

    
    agent_controller_kwargs = {
        "task": task,
        "tools_retreiver": weaviate_controller.retreive,
        "collection"     : request.app.state.collection
    }

    agent_controller   = AgentController(
        feature_id      = feature_id,
        agents          = request.app.state.agents[feature_id],
        client          = request.app.state.groq_client,
        **agent_controller_kwargs
    )


    # read data
    job_title = data.job_title
    job_desc  = data.job_description


    # preprocessing
    try:
        preprocessed = agent_controller.preprocess_input(input = (job_title, job_desc))

    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_PREPROCESSING_INPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_tools_recommendation(message = m)


    # calling the agent
    try:
        agent_output = agent_controller.call_agent(input = preprocessed)
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_tools_recommendation(message = m)
    

    # post-processing
    try:
        agent_output = agent_controller.postprocess_agent_output(agent_output = agent_output)
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_POSTPROCESSING_OUTPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_tools_recommendation(message = m)
    

    # log result
    end_time = perf_counter()
    duration_s = end_time - start_time

    try:
        result_to_log = AgentInferenceResult(
            user_input   = (job_title, job_desc),
            task         = task,
            agent_output = agent_output,
            duration_s   = duration_s,
        )

        feature_controller.log_result(result = result_to_log)
    
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_LOGGING_THE_RESULT.value
        F.print_error(error = e, message = m)
        return get_bad_request_tools_recommendation(message = m)


    return get_good_request_tools_recommendation(
        message = ResponsesEnum.JD_ENH_SUCCESS_TOOLS_RECOMMENDATION.value,
        tools   = agent_output
    )





# Job Description Enhancement
@job_description_enhancement_router.post("/{feature_id}/job_description_enhancement")
async def enhance_job_desc(
    feature_id: str,
    data      : JobEnhancementIP,
    request   : Request
) -> dict[str, bool]:
    """
    This endpoint does the following:
        - validate the given job data
        - Enhance the Job Description

    Returns:
        {
            "success"                  : True, if no error,
            "message"                  : message returned,
            "enhanced_job_description" : if the job desc contain tools or not,
        }
    """
    task = "job_description_enhancement"

    # setup
    start_time = perf_counter()

    if not F.validate_feature_id(feature_id = feature_id):
        return get_bad_request_job_desc_enhancement(message = ResponsesEnum.GENERAL_ERROR_WRONG_FEATURE_ID.value)
    
    # controllers
    feature_controller = FeatureController(feature_id = feature_id)


    agent_controller_kwargs = {
        "task": task
    }

    agent_controller   = AgentController(
        feature_id      = feature_id,
        agents          = request.app.state.agents[feature_id],
        client          = request.app.state.groq_client,
        **agent_controller_kwargs
    )


    # read data
    job_title = data.job_title
    job_desc  = data.job_description
    tools     = data.tools


    # preprocessing
    try:
        preprocessed = agent_controller.preprocess_input(input = (job_title, job_desc, tools))

    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_PREPROCESSING_INPUT.value
        F.print_error(error = e, message = m)
        return get_bad_request_job_desc_enhancement(message = m)


    # calling the agent
    try:
        if tools is not None:
            is_rag_used = True

        else:
            is_rag_used = False
        
        agent_output = agent_controller.call_agent(input = (preprocessed, is_rag_used))
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_CALLING_AGENT.value
        F.print_error(error = e, message = m)
        return get_bad_request_job_desc_enhancement(message = m)
    

    # log result
    end_time = perf_counter()
    duration_s = end_time - start_time

    try:
        result_to_log = AgentInferenceResult(
            user_input   = (job_title, job_desc, tools),
            task         = task,
            agent_output = agent_output,
            duration_s   = duration_s,
        )

        feature_controller.log_result(result = result_to_log)
    
    except Exception as e:
        m = ErrorsEnum.DEBUG_ERROR_LOGGING_THE_RESULT.value
        F.print_error(error = e, message = m)
        return get_bad_request_job_desc_enhancement(message = m)


    return get_good_request_job_desc_enhancement(
        message                  = ResponsesEnum.JD_ENH_SUCCESS_JOB_DESCRIPTION_ENHANCEMENT.value,
        enhanced_job_description = agent_output
    )