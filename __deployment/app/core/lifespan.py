# -------------------------------------------------
# Important FastAPI Utils
# -------------------------------------------------
import os
from time import perf_counter

from fastapi    import FastAPI
from contextlib import asynccontextmanager

from helpers.settings     import get_settings
from helpers.functional import print_success_message, print_error, print_title, print_subtitle

from models.config.agents_config import JOB_DESCRIPTION_ENHANCEMENT_MODELS
from models.config.system_tasks  import (
    FEATURE_IDENITY_RECOGNITION,
    FEATURE_JOB_DESCRIPTION_ENHANCEMENT,
    FEATURE_JOB_RECOMMENDATION_SYSTEM,
    FEATURE_PROFILE_ANALYSIS,
    FEATURE_PROPOSAL_REJECTION_REASONS
)



@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_start = perf_counter()

    try:
        app_name = get_settings().APP_NAME
        app_vers = get_settings().APP_VERSION

        print_title("Starting the App")
        print(f">> APP    : {app_name}")
        print(f">> Version: {app_vers}")
        print()

        # --------------- Dependecies ------------------- #
        try:
            print_subtitle("Import Startup Dependencies")
            from core.loading_clients_agents import get_identity_recognizer, get_retina_face_detector
            from core.loading_clients_agents import get_weaviate_client
            from core.loading_clients_agents import load_proposal_rejection_reasons_agents
            from core.loading_clients_agents import load_profile_scorer_agents
            from core.loading_clients_agents import get_embedding_model, get_raranker, get_weaviate_collection
            from core.loading_clients_agents import get_rs_embedding_engine
            from agents.job_description_enhancement import get_groq_client
            print_success_message("Startup Dependencies Imported Successfully")
        except Exception as e:
            print_error(message = "Error Importing Startup Dependencies", error = e)
            raise RuntimeError("Failed to import startup dependencies") from e

        # --------------- Weaviate ------------------- #
        try:
            print_subtitle("Init Weaviate Client")
            w_client = get_weaviate_client()
            app.state.weaviate_client = w_client
            print_success_message("Weaviate Client Initiated Successfully")
        except Exception as e:
            print_error(message = "Error Initiating Weaviate Client", error = e)
            raise RuntimeError("Failed to initiate Weaviate client") from e
        
        try:
            print_subtitle("Load Weaviate Collection")
            app.state.collection = get_weaviate_collection(client = w_client)
            print_success_message("Weaviate Collection Loaded Successfully")
        except Exception as e:
            print_error(message = "Error Loading the Collection", error = e)
            raise RuntimeError("Failed to load Weaviate collection") from e

        # --------------- Groq Client ------------------- #
        try:
            print_subtitle("Init Groq Client")
            app.state.groq_client = get_groq_client()
            print_success_message("Groq Client Initiated Successfully")

            print_subtitle("Init Google Client")
            settings = get_settings()
            os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
            print_success_message("Google Client Initiated Successfully")

        except Exception as e:
            print_error(message = "Error Initiating Groq Client", error = e)
            raise RuntimeError("Failed to initiate Groq client") from e

        # --------------- Agents ------------------- #
        try:
            print_subtitle("Load Agents")
            app.state.agents = {
                FEATURE_IDENITY_RECOGNITION: {
                    "detector": get_retina_face_detector(),
                    "verifier": get_identity_recognizer()
                },

                FEATURE_JOB_DESCRIPTION_ENHANCEMENT: {
                    "tools_detector"   : JOB_DESCRIPTION_ENHANCEMENT_MODELS["tools_detector"],
                    "tools_recommender": JOB_DESCRIPTION_ENHANCEMENT_MODELS["tools_recommender"],
                    "job_desc_enhancer": JOB_DESCRIPTION_ENHANCEMENT_MODELS["job_desc_enhancer"],
                    "RAG_embedder"     : get_embedding_model(),
                    "RAG_reranker"     : get_raranker(),
                },

                FEATURE_PROFILE_ANALYSIS: load_profile_scorer_agents(),
                FEATURE_JOB_RECOMMENDATION_SYSTEM: {
                    "RS_embedder": get_rs_embedding_engine(),
                },

               FEATURE_PROPOSAL_REJECTION_REASONS: load_proposal_rejection_reasons_agents(),
            }

            print_success_message("Agents Loaded Successfully")

        except Exception as e:
            print_error(message = "Error Loading Agents", error = e)
            raise RuntimeError("Failed to load agents") from e

        startup_duration_s = perf_counter() - startup_start


        print()
        print_success_message("App Started Successfully")
        print_success_message(f"Startup Time: {startup_duration_s:.2f}s")
        print_title(100 * "=")

        yield

    finally:
        print_title("Terminating the App")
        print_subtitle("Releasing Resources")

        if getattr(app.state, "weaviate_client", None) is not None:
            app.state.weaviate_client.close()

        app.state.groq_client = None
        app.state.collection = None
        app.state.agents = None

        print_success_message("App Terminated Successfully")
        print(100 * "=")
