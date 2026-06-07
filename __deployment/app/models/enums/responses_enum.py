from enum import Enum

class ResponsesEnum(Enum):
    # general
    GENERAL_ERROR_WRONG_FEATURE_ID = "Feature ID is not valid."
  
    # identity recognition [ID_RECO]
    ID_RECO_ERROR_LOADING_IMAGES_ERROR = "Error while loading images. Please try again later."
    ID_RECO_ERROR_REQUIRED_HIGH_QUALITY_IMAGE = "Please, upload higher quality images"
    ID_RECO_SUCCESS_PERSON_NOT_VERIFIED = "Please Upload two higher quality images for your personality"
    ID_RECO_SUCCESS_PERSON_VERIFIED = "Successfully verified"

    # job desc enhancement [JD_ENH]
    JD_ENH_SUCCESS_TOOLS_DETECTION             = "Tools Detection Process Completed"
    JD_ENH_SUCCESS_TOOLS_RECOMMENDATION        = "Tools Recommendation Process Completed"
    JD_ENH_SUCCESS_JOB_DESCRIPTION_ENHANCEMENT = "Tools Recommendation Process Completed"
    
    