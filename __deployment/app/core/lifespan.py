# -------------------------------------------------
# Important FastAPI Utils
# -------------------------------------------------

from fastapi import FastAPI
from contextlib import asynccontextmanager

from helpers.functional import get_identity_recognizer, get_retina_face_detector
from helpers.functional import print_success_message, print_error_message

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    - wake up agents - Init databases when the app starts
    - close them when the app ends
    """
    # wake up agents
    print(100 * '=')
    try:
        app.state.agents = {
            "identity_recognition": {
                "detector": get_retina_face_detector(),
                "verifier": get_identity_recognizer()
            }
        }

        print_success_message("Agents Loaded Successfully")
    
    except Exception as e:
        print_error_message(f"Error Loading Agents: {e}")


    yield
    app.state.agents = None
    print(100 * '=')