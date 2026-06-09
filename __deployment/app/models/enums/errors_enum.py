from enum import Enum

class ErrorsEnum(Enum):
    JD_ENH_ERROR_TASK = "Job Description Task is invalid"
    PRR_ERROR_TASK  = "Proposa Rejection Task is Invalid"

    # log results
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