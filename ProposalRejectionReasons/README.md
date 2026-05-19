# GoFreelance: AI System for Proposal Rejection Reasons
A comprehensive system designed to help freelancers understand why their proposals are rejected by clients. It leverages an agentic AI framework based on a multi-agent (sub-agent) architecture to analyze both the client’s job requirements and the freelancer’s proposal. The system identifies the key reasons behind rejection and provides targeted recommendations to improve skills and increase the likelihood of success in future opportunities.


## System Description
The system is designed to evaluate the **job–proposal match**, which measures how well a freelancer’s proposal aligns with a client’s job posting. It is built on a multi-agent architecture composed of five specialized sub-agents, each responsible for analyzing a distinct aspect of the match:

- Job Understanding Agent: Evaluates whether the freelancer has accurately understood the job and proposed relevant solutions or methods to accomplish it.

- Requirement Coverage Agent: Assesses how many of the client’s stated requirements are explicitly addressed in the freelancer’s proposal.

- Tools Alignment Agent: Examines whether the freelancer mentions the tools, technologies, or platforms specified by the client in the job post.

- Evidence of Experience Agent: Determines whether the proposal includes references to prior work or experience relevant to the job.

- Language Clarity Agent: Analyzes the clarity, professionalism, and correctness of the proposal’s language, including misleading phrasing or grammatical issues.

A SuperAgent coordinates the workflow among these sub-agents by managing their outputs and aggregating their findings. It then synthesizes the results into a final assessment and generates personalized recommendations to help the freelancer improve future proposals and increase their chances of being accepted.


## Workflow

### 1.0 Tools Alignment Agent

- Job Tools Extraction: Utilizing an LLM to extract the tools mentioned in the job description. Each tool will contain:
    - tool name
    - tool necessity level: whether the tool was mandatory, recommended, optional, or even forbidden.
    

---


## Requirements

- Python 3.10 or later
- A supported LLM provider API key
- Python virtual environment recommended

## setup
### 1. Clone the repository

```bash
git clone <repo-url>
cd Proposal-Rejection-Reasons
```


### 2. Create a virtual environment (Recommended)

* windows
```bash 
python -m venv venv
venv\Scripts\activate
```

* Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the environment file
copy the example environment file:
```bash
cp .env.example .env
```
Then add the required API keys and configuration values inside `.env`.

### 5. Run the app
```bash
python main.py
```
---