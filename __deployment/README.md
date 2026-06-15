# Go Freelance: AI Models Deployment
This repositry is for developing a centered orchestration FastAPI application for deployment AI models employed in the platform. The use of a single centered app is to simplify the integration with the backend team as a single serviced the platform needs to use for all AI models.

## Project Structure
The project is structured & organized in the following structure:
- `routes`     : the main endpoints used to serve the models.
- `controllers`: logic controlling files organizing, features logging, and agents calling.
- `models`     : schemas defining data objects, requests, and responses.
- `helpers`    : helpful configurations & utility functions.
- `assets`     : saved results & visualizations
- ` main.py`   : main driver


## Routes Specifications
### Base Route
- Used for testing if the system works.
- Accessed by the base endpoint path: 
bash
```
/host/ai/api
```


- No inputs required.
- Output

```bash
{
    APP Name: str,
    APP Version: str
}
```

Note: The base endpoint path is a prefix that should be used before any other route.

---

### Identity Recognition Route
- Used to verify uploaded person images.
- Request Expected:
bash
```
POST base_endpoint_path/identity_recognition/verify_images
Content-Type: multipart/form-data

img1: first image file
img2: second image file
```

- Outputs:

on verified
```bash
{
    "success"             : True,
    "message"             : verified message [to user],
    "verified"            : True,
    "similarity"          : similarity between the two images,
    "similarity_threshold": the threshold used to determine the verified vs not-verified,
    "person_embeddings"   : person_embeddings to save for future access
}
```

on not-verified
```bash
{
    "success"             : True,
    "message"             : not-verified message [to user],
    "verified"            : False,
    "similarity"          : similarity between the two images,
    "similarity_threshold": the threshold used to determine the verified vs not-verified,
    "person_embeddings"   : None
}
```
on failure
```bash
{
    "success"             : False,
    "message"             : error message [to the system],
    "verified"            : None,
    "similarity"          : None,
    "similarity_threshold": None,
    "person_embeddings"   : None
}
```

---

### Job Description Enhancement Route
#### Task1: Tools Detection
- Used to detect tools in the client job description.
- Request Expected:
bash
```
POST base_endpoint_path/job_description_enhancement/tools_detection
Content-Type: application/json

{
  "job_description": str
}
```

- Outputs:

on success
```bash
{
    "success"   : True,
    "message"   : success message [to system],
    "has_tools" : boolean indicates if the client job description has tools or not.
}
```

on failure
```bash
{
    "success"   : False,
    "message"   : error message [to system],
    "has_tools" : None
}
```

#### Task2: Tools Recommendation
- Used to recommend tools to the client.
- Request Expected:
bash
```
POST base_endpoint_path/job_description_enhancement/tools_recommendation
Content-Type: application/json

{
    "job_tilte": str
    "job_description": str
}
```

on error
```bash
{
    "success"             : False,
    "message"             : error message [to the system],
    "verified"            : None,
    "similarity"          : None,
    "similarity_threshold": None,
    "person_embeddings"   : None
}
```

## Setup
#### Install Python 3.10 or later.

#### Creating / activating a virtal environment (recommended)
1. Windows
```bash
python -m venv venv_name
\venv_dir\venv_name\Scripts\activate
```

2. Linux-based Systems
```bash
python3 -m venv venv_name
source venv_dir/venv_name/bin/activate
```

#### Installing Requirements
```bash
pip install -r requirements.txt
```

#### Creating a `.env` file 
```bash
cp .env.example .env
```
Fill it with the required values


#### Running the App
```bash 
uvicorn main:app --reload
```