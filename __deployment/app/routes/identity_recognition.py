# ----------------------------------------------
# Serving Identity Recognition 
# ----------------------------------------------

# helpers
from helpers.config import ROUTE_MAIN_ROUTE
import helpers.functional as F
from time import perf_counter

# messages
from models.message_enums import IdentityRecognitionMessages
from models.message_enums import ResponsesEnum
from models.pydantic_schemas import AgentInferenceResult, ImageLog

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
    """Return a bad request error specific for identity recognition api"""
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
    """Return a good request specific for identity recognition api"""
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
identity_recognition_router = APIRouter(
    prefix = ROUTE_MAIN_ROUTE 
)

@identity_recognition_router.post("/{feature_id}/verify_images")
async def verify_person_images(
    feature_id: str,
    request   : Request,
    img1      : UploadFile = File(...),
    img2      : UploadFile = File(...)
) -> dict[str, bool | list[float] | str]:
    """
    This endpoint does the following:
        - Read the uploaded images
        - pre-process the images to prepare them for the model
        - feed the images to the model
        - format the results

    Returns:
        {
            "success"  : true if success else false.
            "verified" : true if the same person, false if not, or None if success = False
            "person_embeddings": person face embeddings if verified
            "message"  : message returned
        }
    """
    # setup
    start_time = perf_counter()
    if not F.validate_feature_id(feature_id = feature_id):
        return return_bad_request(message = ResponsesEnum.ERROR_WRONG_FEATURE_ID.value)

    # controllers
    feature_controller = FeatureController(feature_id = feature_id)
    agent_controller = AgentController(
        feature_id = feature_id,
        agents     = request.app.state.agents[feature_id]
    )


    # read images
    img1_log = ImageLog(
        filename     = str(img1.filename),
        content_type = str(img1.content_type)
    )

    img2_log = ImageLog(
        filename     = str(img2.filename),
        content_type = str(img2.content_type)
    )

    try:
        img1 = await img1.read()
        img2 = await img2.read()
    except Exception as e:
        F.print_error_message(e)
        return return_bad_request(message = IdentityRecognitionMessages.ERROR_LOADING_IMAGES_ERROR.value)
    
    img1_log.size_mbytes = len(img1) / (1024 * 1024)
    img2_log.size_mbytes = len(img2) / (1024 * 1024)
    
    # preprocess
    try:
        preprocessed = agent_controller.preprocess_input(input = (img1, img2))
    except Exception as e:
        F.print_error_message(e)

    faces = preprocessed["faces"]


    # calling the agent
    try:
        agent_output = agent_controller.call_agent(input = faces)
    except Exception as e:
        F.print_error_message(e)
    


    # post-process
    try:
        verification_results = agent_controller.postprocess_agent_output(agent_output = agent_output)
    except Exception as e:
        F.print_error_message(e)


    # log result
    end_time = perf_counter()
    duration = end_time - start_time

    try:
        result_to_log = AgentInferenceResult(
            images       = [img1_log, img2_log],
            agent_output = verification_results['verified'],
            duration_s   = duration,
        )

        feature_controller.log_result(result = result_to_log)
    
    except Exception as e:
        F.print_error_message(e)


    verified = verification_results["verified"]
    if not verified:
        return return_good_request(
            message = IdentityRecognitionMessages.SUCCESS_PERSON_NOT_VERIFIED.value,
            verification_results = verification_results
        )
 

    return return_good_request(
        message = IdentityRecognitionMessages.SUCCESS_PERSON_VERIFIED.value,
        verification_results = verification_results
    )

