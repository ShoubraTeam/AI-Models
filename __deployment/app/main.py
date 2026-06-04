# ----------------------------------------
# Main Driver File
# ----------------------------------------


from fastapi import FastAPI
from core.lifespan import lifespan


# routers
from routes.base                import base_router
from routes.identity_recognition import identity_recognition_router



app = FastAPI(lifespan = lifespan)


# routers
app.include_router(router = base_router)
app.include_router(router = identity_recognition_router)