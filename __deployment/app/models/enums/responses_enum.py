# ---------------------------------------------------
# Responses sent to the client
# ---------------------------------------------------


from enum import Enum

class ResponsesEnum(Enum):  
    # identity recognition [ID_RECO]
    ID_RECO_ERROR_LOADING_IMAGES_ERROR        = "Error while loading images. Please try again later."
    ID_RECO_ERROR_REQUIRED_HIGH_QUALITY_IMAGE = "Please, upload higher quality images"
    ID_RECO_SUCCESS_PERSON_NOT_VERIFIED       = "Please Upload two higher quality images for your personality"
    ID_RECO_SUCCESS_PERSON_VERIFIED           = "Successfully verified"
    ID_RECO_NO_CARD                           = "At least one image shoud be an identity card"
    ID_RECO_NO_PERSONAL                       = "At least one image shoud be a personal card"

    # job desc enhancement [JD_ENH]
    JD_ENH_SUCCESS_TOOLS_DETECTION             = "Tools Detection Process Completed"
    JD_ENH_SUCCESS_TOOLS_RECOMMENDATION        = "Tools Recommendation Process Completed"
    JD_ENH_SUCCESS_JOB_DESCRIPTION_ENHANCEMENT = "Job Enhancement Process Completed"


    # PRR
    PRR_JOB_FEATURES_EXTRACTED_CORRECTLY = "Job Features Extracted Correctly"
    PRR_PROPOSAL_ANALYSIS_COMPLETED      = "Proposal Analysis Process Completed"
    
    # Profile Scorer 
    PROFILE_SCORER_FEATURES_EXTRACTED_CORRECTLY = "Profile Features Extracted Correctly"
    PROFILE_SCORER_ANALYSIS_COMPLETED           = "Profile Analysis Process Completed"
    PROFILE_SCORER_IMAGE_LOADING_ERROR          = "Error while loading images. Please try again later."