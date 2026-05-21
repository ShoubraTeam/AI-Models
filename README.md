# GoFreelance: AI System for Proposal Rejection Reasons
A comprehensive system designed to help freelancers understand why their proposals are rejected by clients. It leverages an agentic AI framework based on a multi-agent (sub-agent) architecture to analyze both the client's job requirements and the freelancer's proposal. The system identifies the key reasons behind rejection and provides targeted recommendations to improve skills and increase the likelihood of success in future opportunities.


## System Description
The system is designed to evaluate the **job–proposal match**, which measures how well a freelancer's proposal aligns with a client's job posting. It is built on a multi-agent architecture composed of five specialized sub-agents, each responsible for analyzing a distinct aspect of the match:

- **Job Understanding Agent**: Evaluates whether the freelancer has accurately understood the job and proposed relevant solutions or methods to accomplish it.

- **Requirement Coverage Agent**: Assesses how many of the client's stated requirements are explicitly addressed in the freelancer's proposal.

- **Tools Alignment Agent**: Examines whether the freelancer mentions the tools, technologies, or platforms specified by the client in the job post.

- **Evidence of Experience Agent**: Determines whether the proposal includes references to prior work or experience relevant to the job.

- **Language Clarity Agent**: Analyzes the clarity, professionalism, and correctness of the proposal's language, including misleading phrasing or grammatical issues.

A SuperAgent coordinates the workflow among these sub-agents by managing their outputs and aggregating their findings. It then synthesizes the results into a final assessment and generates personalized recommendations to help the freelancer improve future proposals and increase their chances of being accepted.


## Workflow

### 1.0 Tools Alignment

- **Job Tools Extraction**: Utilizing an LLM to extract the tools mentioned in the job description. Each tool will contain:
    - tool name
    - tool necessity level: whether the tool was mandatory, recommended, optional, or even forbidden.

- **Proposal Analysis**: Utilizing an LLM to analyze the proposal & return a report for each tool extracted from the job description. The report will contain:
    - tool name & tool necessity level as found in the job description.
    - found in proposal: a boolean variable indicating if the tool was mentioned in the proposal or not.
    - with confidence: a boolean variable indicating if the freelancer mentioned the tool with confidence or it was just generic mentioning. The agent outputs that by analyzing context.

- **Scoring**: A scoring mechanism calculates the ratio of the tools mentioned in the proposal weighted by the necessity level & confidence.

---

### 2.0 Requirement Coverage Agent

- **Job Requirements Extraction**: Utilizes the `JobRequirementsExtractor` sub-agent to parse the raw job description text and compile an atomic, clean list of core functional deliverables and timeline constraints (strictly excluding developer tools or languages to avoid architectural redundancy).
- **Proposal Semantic Matching**: Employs the `JobRequirementsMatcher` sub-agent to map the extracted requirements against the freelancer's proposal. It applies a strict validation evaluation to penalize missing features or timeline violations, preventing False Positives.
- **Pipeline Orchestration**: The `RequirementCoverageAgent` coordinates the sequential execution flow, formatting and passing the extraction data directly into the matcher, returning a fully validated Pydantic data object (`score`, `details`, `client_requirements`, `requirements_covered`, `missing_requirements`).

---

### 3.0 Job Understanding Agent

- **Job Key Points Extraction**: Utilizes the `JobKeyPointsExtractor` sub-agent to parse the job description and extract:
    - core problem: the main goal or problem the client wants to solve.
    - required deliverables: the concrete outcomes the client expects.
    - key keywords: domain-specific tools, skills, and technologies mentioned in the job.

- **Proposal Evaluation**: Employs the `JobUnderstandingEvaluator` sub-agent to assess the proposal against the extracted key points. It checks whether the freelancer identified the core problem, proposed a concrete solution, mentioned practical steps, and covered the key keywords. The evaluator returns:
    - score (0–10): how well the freelancer understood the job.
    - matched / missing keywords: which key terms appeared or were absent in the proposal.
    - boolean flags: problem identified, solution proposed, practical steps mentioned.
    - irrelevant content: any off-topic content found in the proposal.
    - summary: a short explanation of the score.
    - confidence score: how confident the agent is in its evaluation.

- **Rule-Based Decision**: A scoring threshold (default: 5.0/10) determines whether the proposal passes or fails this dimension. Proposals that fall below the threshold receive a rejection reason generated from the evaluation summary.

- **Pipeline Orchestration**: The `JobUnderstandingAgent` coordinates the sequential execution of both sub-agents, passing the extracted key points directly into the evaluator and returning a fully validated Pydantic data object.

---

## Requirements

- Python 3.10 or later
- A supported LLM provider API key
- Python virtual environment recommended

## Setup
### 1. Clone the repository

```bash
git clone <repo-url>
cd Proposal-Rejection-Reasons
```

### 2. Create a virtual environment (Recommended)

* Windows
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
Copy the example environment file:
```bash
cp .env.example .env
```
Then add the required API keys and configuration values inside `.env`.

### 5. Run the app
```bash
cd app
python main.py
```
