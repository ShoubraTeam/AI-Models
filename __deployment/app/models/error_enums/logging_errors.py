from enum import Enum

class LoggingErrors(Enum):
    JSON_FILE_LOADING_ERROR    = "Json file loading error"
    JSON_FILE_SAVING_ERROR     = "Json file saving error"
    JSON_FILE_PROCESSING_ERROR = "Json file processing error"