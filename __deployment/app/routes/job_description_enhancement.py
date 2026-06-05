# ----------------------------------------------
# Job Desc Enhancement
# ----------------------------------------------

# helpers
from helpers.config import ROUTE_MAIN_ROUTE
import helpers.functional as F
from time import perf_counter

# messages
from models.message_enums    import JobDescriptionEnhancementMessages
from models.message_enums    import ResponsesEnum
from models.pydantic_schemas import AgentInferenceResult
from models.pydantic_schemas import JobEnhancementIP, ToolsDetectionIP

# controllers
from controllers import FeatureController
from controllers import AgentController

# fast api
from fastapi import APIRouter, Request
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import status


# -------------------------- Helper Functions ---------------------------
def return_bad_request(message: str) -> JSONResponse:
    """Return a bad request error specific for job_desc_enhancement api"""
    return JSONResponse(
        status_code = status.HTTP_400_BAD_REQUEST,
        content = {
            "success"  : False,
            "verified" : None,
            "person_embeddings": None,
            "message"  : message
        }
    )

def return_good_request(message: str, verification_results: dict[str, bool | float | list[float]]) -> JSONResponse:
    """Return a good request specific for job_desc_enhancement api"""
    verified = verification_results['verified']
    similarity = verification_results["similarity"]
    similarity_threshold = verification_results['similarity_threshold']
    person_embeddings = verification_results["person_embeddings"]
    
    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "success"             : True,
            "message"             : message,
            "verified"            : verified,
            "similarity"          : similarity,
            "similarity_threshold": similarity_threshold,
            "person_embeddings"   : person_embeddings,
        }
    )


# -------------------------------- Routing ---------------------------------
job_description_enhancement_router = APIRouter(
    prefix = ROUTE_MAIN_ROUTE 
)


@job_description_enhancement_router.post("/{feature_id}/verify_images")
async def verify_person_images(
    feature_id: str,
    request   : Request,
) -> dict[str, bool | list[float] | str]:
    
    start_time = perf_counter()
    pass

@job_description_enhancement_router.post("/{feature_id}/detect_tools")
async def detect_tools(
    feature_id: str,
    data      : ToolsDetectionIP,
    request   : Request
):
    """
    This endpoint does the following:
        - validate the given job data
        - detecting tools in the desc
        - If there are tools -> return them
        - If there are not   -> return tools relevant to the job

    Returns:
        {
            
        }
    """
    # setup
    start_time = perf_counter()
    print(100 * '=')
    print(start_time)

    if not F.validate_feature_id(feature_id = feature_id):
        return return_bad_request(message = ResponsesEnum.ERROR_WRONG_FEATURE_ID.value)
    
    # controllers
    feature_controller = FeatureController(feature_id = feature_id)
    agent_controller   = AgentController(
        feature_id = feature_id,
        agents     = request.app.state.agents[feature_id],
        client     = request.app.state.groq_client
    )    


    # read data
    job_title = data.job_title
    job_desc  = data.job_description


    # preprocess
    try:
        preprocessed = agent_controller.preprocess_input(
            input  = (job_title, job_desc),
            kwargs = {"task": "detect_tools"}
        )

    except Exception as e:
        F.print_error_message(e)

    print(preprocessed)

    return {

    }
    


    # # detect tools
    # has_tools = enhnacer.detect_tools(job_desc = job_desc)
    # if has_tools:
    #     return {
    #         'status'         : 'success',
    #         'has_tools'      : 1,
    #         'suggested_tools': None
    #     }
    
    # else:
    #     tools = enhnacer.get_relevant_tools(job_title = job_title, job_desc = job_desc, max_retries = 3)
    #     return {
    #         'status'         : 'success',
    #         'has_tools'      : 0,
    #         'suggested_tools': tools
    #     }