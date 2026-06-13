# ----------------------------------------
# Main Driver File
# ----------------------------------------

import dotenv
dotenv.load_dotenv()

# FastAPI utils
from fastapi import FastAPI
from core.startup_noise import configure_startup_noise

configure_startup_noise()

from core.lifespan import lifespan

# routers
from routes.base                        import base_router
from routes.identity_recognition        import identity_recognition_router
from routes.job_description_enhancement import job_description_enhancement_router
from routes.proposal_rejection_reasons  import proposal_rejection_reasons_router
from routes.profile_analysis import profile_analysis_router

from routes.job_recommendation_system import job_recommendation_system_router

app = FastAPI(lifespan = lifespan)


# routers
# app.include_router(router = base_router)
app.include_router(router = identity_recognition_router)
# app.include_router(router = job_description_enhancement_router)
# app.include_router(router = proposal_rejection_reasons_router)
# app.include_router(router = profile_analysis_router)
# app.include_router(router = job_recommendation_system_router)
