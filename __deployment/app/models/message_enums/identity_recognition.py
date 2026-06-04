# --------------------------------------------------------
# Enum for displaying Identity Recognition Messages
# --------------------------------------------------------

from enum import Enum

class IdentityRecognitionMessages(Enum):
    ERROR_LOADING_IMAGES_ERROR = "Error while loading images. Please try again later."
    ERROR_REQUIRED_HIGH_QUALITY_IMAGE = "Please, upload higher quality images"

    SUCCESS_PERSON_NOT_VERIFIED = "Please Upload two higher quality images for your personality"
    SUCCESS_PERSON_VERIFIED = "Successfully verified"
