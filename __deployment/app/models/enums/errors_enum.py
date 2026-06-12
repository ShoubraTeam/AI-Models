# --------------------------------------------- #
# Errors that should help debug the system
# --------------------------------------------- #


from enum import Enum

class ErrorsEnum(Enum):
    GENERAL_Invalid_FEATURE_ID = "Feature ID is not valid."

    # task errors
    JD_ENH_INVALID_TASK = "Job Description Task is invalid"
    PRR_INVALID_TASK    = "Proposal Rejection Reasons Task is Invalid"
    ID_INVALID_TASK     = "Identity Recognition Task is Invalid"
    RS_INVALID_TASK     = "Job Recommendation System Task is Invalid"
    PS_INVALID_TASK     = "Profile Analysis Task is Invalid"
    

    # log errors
    LOG_JSON_FILE_LOADING_ERROR    = "Json file loading error"
    LOG_JSON_FILE_SAVING_ERROR     = "Json file saving error"
    LOG_JSON_FILE_PROCESSING_ERROR = "Json file processing error"

    # debug
    DEBUG_ERROR_LOADING_DATA             = "Error while loading data: "
    DEBUG_ERROR_PREPROCESSING_INPUT      = "Error while Preprocessing:"
    DEBUG_ERROR_CALLING_AGENT            = "Error while Calling Agent:"
    DEBUG_ERROR_POSTPROCESSING_OUTPUT    = "Error while Postprocessing:"
    DEBUG_ERROR_LOGGING_THE_RESULT       = "Error while Logging Result:"
    DEBUG_WEAVIATE_BUILD_ERROR           = "Error Building Weaviate Collection"
    DEBUG_WEAVIATE_EMPTY_DATA            = "Data is Empty"