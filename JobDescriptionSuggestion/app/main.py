# ------------------------------------------
# FastAPI server for serving ASync AI Calls
# ------------------------------------------

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.schemas import ToolsDetectionIP, JobEnhancementIP
import src.utils.config as CFG
from src.job_enhancer import Enhancer
from dotenv import load_dotenv

# cfg
app = FastAPI()
app.mount("/static", StaticFiles(directory = "app/static"), name = "static")
templates = Jinja2Templates(directory = "app/templates")
load_dotenv()

# the enhancer
enhnacer = Enhancer(
    enhancement_model = CFG.ENHANCEMENT_MODEL_1,
    detection_model = CFG.DETECTION_MODEL,
    skills_extractor = CFG.SKILLS_EXTRACTOR_MODEL,
    collection_name = CFG.COLLECTION_NAME,
    model_provider = "groq"
)


# index.html
@app.get("/")
async def render_index(request: Request):
    return templates.TemplateResponse(name = "index.html", request = request)

# detecting tools
@app.post(path = "/detect_tools")
async def detect_tools(data: ToolsDetectionIP):
    # get data
    job_title = data.job_title
    job_desc = data.job_desc

    if not job_desc or not job_title:
        return {
            'status'         : 'error',
            'has_tools'      : 0,
            'suggested_tools': None
        }

    # detect tools
    has_tools = enhnacer.detect_tools(job_desc = job_desc)
    if has_tools:
        return {
            'status'         : 'success',
            'has_tools'      : 1,
            'suggested_tools': None
        }
    
    else:
        tools = enhnacer.get_relevant_tools(job_title = job_title, job_desc = job_desc, max_retries = 3)
        return {
            'status'         : 'success',
            'has_tools'      : 0,
            'suggested_tools': tools
        }



# enhancing
@app.post(path = "/enhance_desc")
async def enhance_job_desc(data: JobEnhancementIP):
    # get data
    job_title = data.job_title
    job_desc = data.job_desc
    tools = data.tools    

    if not job_desc or not job_title:
        return {
            'status'       : 'error',
            'original_desc': None,
            'enhanced_desc': None
        }
    
    # enhance
    enhanced = enhnacer.enhnace(
        job_title = job_title,
        job_desc = job_desc,
        suggested_tools = tools,
    )

    return {
        'status'       : 'success',
        'original_desc': job_desc,
        'enhanced_desc': enhanced
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host = "0.0.0.0", 
        port = 8000,
        reload = True
    )


    