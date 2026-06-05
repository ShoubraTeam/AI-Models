# -------------------------------------------------
# Important FastAPI Utils
# -------------------------------------------------

from fastapi import FastAPI
from contextlib import asynccontextmanager

from helpers.functional import get_identity_recognizer, get_retina_face_detector
from helpers.functional import print_success_message, print_error_message
from helpers.config     import JOB_DESCRIPTION_ENHANCEMENT_MODELS


from agents.job_description_enhancement import get_groq_client

from models.data_config import (
    FEATURE_IDENITY_RECOGNITION,
    FEATURE_JOB_DESCRIPTION_ENHANCEMENT,
    FEATURE_JOB_RECOMMENDATION_SYSTEM,
    FEATURE_PROFILE_ANALYSIS,
    FEATURE_PROPOSAL_REJECTION_REASONS
)


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
            FEATURE_IDENITY_RECOGNITION: {
                # "detector": get_retina_face_detector(),
                # "verifier": get_identity_recognizer()
            },

            FEATURE_JOB_DESCRIPTION_ENHANCEMENT: {
               "tools_detector"   : JOB_DESCRIPTION_ENHANCEMENT_MODELS["tools_detector"],
               "tools_recommender": JOB_DESCRIPTION_ENHANCEMENT_MODELS["tools_recommender"],
               "job_desc_enhancer": JOB_DESCRIPTION_ENHANCEMENT_MODELS["job_desc_enhancer"],
            },

            FEATURE_JOB_RECOMMENDATION_SYSTEM: {
                
            },

            FEATURE_PROFILE_ANALYSIS: {
               
            },

            FEATURE_PROPOSAL_REJECTION_REASONS: {
                
            },
        }
        
        print_success_message("Agents Loaded Successfully")

    except Exception as e:
        print_error_message(f"Error Loading Agents:")
        raise Exception(e)

    
    # init clients [Groq | weaviate]
    try:
        app.state.groq_client = get_groq_client()
        print_success_message("Groq Client Initiated Successfully")
    except Exception as e:
        print_error_message(f"Error Initiating Groq Client:")
        raise Exception(e)


    yield
    app.state.agents = None
    print(100 * '=')