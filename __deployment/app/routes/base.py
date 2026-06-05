# --------------------------------
# Base Router for testing the API
# --------------------------------

from helpers.config import ROUTE_MAIN_ROUTE
from helpers.config import get_settings


from fastapi import APIRouter
from fastapi import status
from fastapi.responses import JSONResponse

base_router = APIRouter(
    prefix = ROUTE_MAIN_ROUTE
)


@base_router.get("/")
async def welcome():
    settings = get_settings()

    app_name   = settings.APP_NAME
    app_verion = settings.APP_VERSION

    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "App Name"   : app_name,
            "App Version": app_verion
        }
    )