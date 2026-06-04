# Go Freelance: AI Models Deployment
This repositry is for developing a centered orchestration FastAPI application for deployment AI models employed in the platform. The use of a single centered app is to simplify the integration with the backend team as a single serviced the platform needs to use for all AI models.

## Project Structure
The project is structured & organized in the following structure:
- `routes`     : the main endpoints used to serve the models.
- `controllers`: logic controlling files organizing, features logging, and agents calling.
- `models`     : schemas defining data objects, requests, and responses.
- `helpers`    : helpful configurations & utility functions.
- `assets`     : saved results & visualizations
` main.py`     : main driver



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