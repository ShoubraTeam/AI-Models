# Go Freelance: AI Models Deployment

This repository contains a centralized FastAPI orchestration service for deploying and serving the AI models used by the Go Freelance platform.

The main purpose of this service is to provide the backend team with a single, consistent integration point for all AI-powered features, instead of integrating with each model or agent separately.

The service is responsible for:

* Exposing AI features through well-defined API routes.
* Receiving platform requests from the backend.
* Calling the appropriate AI models, agents, or embedding pipelines.
* Returning structured responses that can be stored, displayed, or used by the platform.
* Supporting logging, monitoring, and future model replacement without changing the main backend integration flow.

---

## Project Structure

```text
.
├── routes/          # API endpoints used to serve each AI feature
├── controllers/     # Business logic, feature orchestration, logging, and agent calls
├── models/          # Pydantic schemas for requests, responses, and internal data objects
├── helpers/         # Configuration files and reusable utility functions
├── assets/          # Saved results, generated files, and visualizations
└── main.py          # FastAPI application entry point
```

---

## Base API Prefix

All routes are served under the following base endpoint path:

```http
/host/ai/api
```

This prefix should be added before all route paths listed in this document.

Example:

```http
/host/ai/api/identity_recognition/verify_images
```

---

## Health Check Route

### Purpose

Used to verify that the AI deployment service is running successfully.

### Endpoint

```http
GET /host/ai/api
```

### Request

No request body is required.

### Response

```json
{
  "app_name": "string",
  "app_version": "string"
}
```

---

# 1. Identity Recognition

## Workflow

```text
person images
    ↓
verification endpoint
    ↓
if verified:
    return verified = true
    return person_embeddings
    backend stores person_embeddings in a vector database
else:
    return verified = false
```

The identity recognition workflow is used to verify whether two uploaded person images belong to the same person.

If the images are verified successfully, the endpoint returns the calculated person embeddings. These embeddings should be stored by the backend in a vector database for future identity matching or verification.

---

## Verify Person Images

### Endpoint

```http
POST /host/ai/api/identity_recognition/verify_images
Content-Type: multipart/form-data
```

### Request Shape

```text
img1: first image file
img2: second image file
```

### Successful Response: Verified

```json
{
  "success": true,
  "message": "verified message to user",
  "verified": true,
  "similarity": 0.0,
  "similarity_threshold": 0.0,
  "person_embeddings": [0.0]
}
```

### Successful Response: Not Verified

```json
{
  "success": true,
  "message": "not-verified message to user",
  "verified": false,
  "similarity": 0.0,
  "similarity_threshold": 0.0,
  "person_embeddings": null
}
```

### Failure Response

```json
{
  "success": false,
  "message": "error message to system",
  "verified": null,
  "similarity": null,
  "similarity_threshold": null,
  "person_embeddings": null
}
```

---

# 2. Job Description Enhancement

The Job Description Enhancement feature helps improve client job descriptions by detecting whether the description already contains technical tools, recommending missing tools when needed, and generating an enhanced job description.

## Workflow

```text
client job description
    ↓
Task 1: Detect tools
    ↓
if tools are found:
    send job description directly to Task 3
    generate enhanced job description

if tools are not found:
    ask client if they want tool recommendations
        ↓
    if client agrees:
        Task 2: Recommend tools
            ↓
        return recommended tools to client
            ↓
        client selects preferred tools
            ↓
        Task 3: Enhance job description using:
            - job title
            - original job description
            - selected tools
```

---

## Task 1: Tools Detection

### Purpose

Detects whether the client job description already contains tools, technologies, frameworks, libraries, or platforms.

### Endpoint

```http
POST /host/ai/api/job_description_enhancement/tools_detection
Content-Type: application/json
```

### Request Shape

```json
{
  "job_description": "string"
}
```

### Success Response

```json
{
  "success": true,
  "message": "success message to system",
  "has_tools": true
}
```

### Failure Response

```json
{
  "success": false,
  "message": "error message to system",
  "has_tools": null
}
```

---

## Task 2: Tools Recommendation

### Purpose

Recommends suitable tools based on the job title and job description.

This task is usually called only when Task 1 detects that the job description does not already contain enough tools.

### Endpoint

```http
POST /host/ai/api/job_description_enhancement/tools_recommendation
Content-Type: application/json
```

### Request Shape

```json
{
  "job_title": "string",
  "job_description": "string"
}
```

### Success Response

```json
{
  "success": true,
  "message": "success message to system",
  "tools": ["string"]
}
```

### Failure Response

```json
{
  "success": false,
  "message": "error message to system",
  "tools": null
}
```

---

## Task 3: Job Description Enhancement

### Purpose

Enhances the client job description and makes it clearer, more professional, and more attractive to qualified freelancers.

### Endpoint

```http
POST /host/ai/api/job_description_enhancement/job_description_enhancement
Content-Type: application/json
```

### Request Shape

```json
{
  "job_title": "string",
  "job_description": "string",
  "tools": ["string"]
}
```

If no tools are selected or required:

```json
{
  "job_title": "string",
  "job_description": "string",
  "tools": null
}
```

### Success Response

```json
{
  "success": true,
  "message": "success message to system",
  "enhanced_job_description": "string"
}
```

### Failure Response

```json
{
  "success": false,
  "message": "error message to system",
  "enhanced_job_description": null
}
```

---

# 3. Job Recommendation System

The Job Recommendation System is responsible for embedding freelancer profiles and job posts, storing them in a vector database, and using vector similarity to recommend the most relevant jobs to freelancers.

## Workflow

```text
freelancer data
    ↓
Task 1: Generate freelancer embedding
    ↓
store freelancer embedding in vector database


job data
    ↓
Task 2: Generate job embedding
    ↓
store job embedding in vector database


when freelancer opens the job dashboard:
    ↓
backend retrieves freelancer embedding
    ↓
backend retrieves available job embeddings
    ↓
system ranks jobs by similarity
    ↓
best matching jobs are displayed to freelancer
```

---

## Task 1: Freelancer Embedding

### Purpose

Creates an enriched text representation of freelancer data and generates an embedding vector that can be stored in a vector database.

### Endpoint

```http
POST /host/ai/api/job_recommendation_system/freelancer_embedding
Content-Type: application/json
```

### Request Shape

```json
{
  "bio": "string",
  "skills": ["string"],
  "job_title": "string"
}
```

### Success Response

```json
{
  "success": true,
  "message": "success message to system",
  "enriched_text": "string",
  "embedding": [0.0],
  "embedding_dim": 0
}
```

### Failure Response

```json
{
  "success": false,
  "message": "error message to system",
  "enriched_text": null,
  "embedding": null,
  "embedding_dim": null
}
```

---

## Task 2: Job Embedding

### Purpose

Creates an enriched text representation of job data and generates an embedding vector that can be stored in a vector database.

### Endpoint

```http
POST /host/ai/api/job_recommendation_system/job_embedding
Content-Type: application/json
```

### Request Shape

```json
{
  "description": "string",
  "job_title": "string",
  "skills": ["string"]
}
```

### Success Response

```json
{
  "success": true,
  "message": "success message to system",
  "enriched_text": "string",
  "embedding": [0.0],
  "embedding_dim": 0
}
```

### Failure Response

```json
{
  "success": false,
  "message": "error message to system",
  "enriched_text": null,
  "embedding": null,
  "embedding_dim": null
}
```

---

# 4. Profile Analysis

The Profile Analysis feature evaluates freelancer profile quality by analyzing numerical profile information, bio content, and declared skills.

## Workflow

```text
freelancer profile data
    ↓
Task 1: Extract profile features
    ↓
backend stores extracted features in database
    ↓
Task 2: Generate final profile analysis
    ↓
return profile report
    ↓
display report to client or freelancer
```

Task 1 can be used independently to extract and store structured profile features.

Task 2 uses the original profile data together with the extracted features to generate the final profile analysis report.

---

## Task 1: Profile Features Extraction

### Purpose

Extracts structured features from a freelancer profile, including numerical indicators, bio quality signals, and skills-related signals.

### Endpoint

```http
POST /host/ai/api/profile_analysis/profile_features_extraction
Content-Type: application/json
```

### Request Shape

```json
{
  "job_role": "string",
  "hourly_rate": 0.0,
  "rating": 0.0,
  "total_completed_jobs": 0,
  "bio_text": "string",
  "declared_skills": ["string"]
}
```

### Success Response

```json
{
  "success": true,
  "message": "message to system",
  "numerical_res": {},
  "bio_res": {},
  "skills_res": {}
}
```

### Failure Response

```json
{
  "success": false,
  "message": "message to system",
  "numerical_res": null,
  "bio_res": null,
  "skills_res": null
}
```

---

## Task 2: Profile Final Analysis

### Purpose

Generates the final profile analysis report based on the freelancer profile data, previously extracted profile features, and the uploaded freelancer profile image.

This endpoint requires both:

* Profile analysis data as a JSON string inside a form field named `data`.
* Profile image file inside a file field named `profile_img`.

### Endpoint

```http
POST /host/ai/api/profile_analysis/{feature_id}/profile_final_analysis
Content-Type: multipart/form-data
```

### Request Shape

```text
data: JSON string containing the profile data and pre-extracted features
profile_img: uploaded profile image file
```

### `data` Field Shape

The `data` field must be sent as a JSON string matching the following structure:

```json
{
  "job_role": "string",
  "hourly_rate": 0.0,
  "rating": 0.0,
  "total_completed_jobs": 0,
  "bio_text": "string",
  "declared_skills": ["string"],
  "pre_extracted_features": {}
}
```

### Example Request Shape

```text
POST /host/ai/api/profile_analysis/profile_scorer/profile_final_analysis
Content-Type: multipart/form-data

data:
{
  "job_role": "Backend Developer",
  "hourly_rate": 20.0,
  "rating": 4.8,
  "total_completed_jobs": 35,
  "bio_text": "Experienced backend developer specialized in FastAPI and PostgreSQL.",
  "declared_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
  "pre_extracted_features": {
    "numerical_res": {},
    "bio_res": {},
    "skills_res": {}
  }
}

profile_img: uploaded image file
```

### Success Response

```json
{
  "success": true,
  "message": "message to system",
  "profile_report": "string"
}
```

### Failure Response

```json
{
  "success": false,
  "message": "message to system",
  "profile_report": null
}
```


---

# 5. Proposal Rejection Reasons

The Proposal Rejection Reasons feature analyzes job posts and freelancer proposals to explain why a proposal may be rejected or considered weak.

It is designed to support fast retrieval by storing both the extracted job features and the generated proposal analysis results.

## Workflow

```text
when a new job is added:
    ↓
Task 1: Extract job features
    ↓
backend stores job features in database


when a proposal is submitted:
    ↓
backend retrieves stored job features
    ↓
Task 2: Analyze proposal
    ↓
backend stores proposal analysis report


when user requests proposal rejection reasons:
    ↓
backend retrieves stored proposal report
    ↓
display report immediately
```

This means the proposal rejection report should usually be generated at proposal submission time, not when the user opens the report page. This improves response speed and user experience.

---

## Task 1: Job Features Extraction

### Purpose

Extracts structured job features from the job description, including tools, requirements, and key points.

### Endpoint

```http
POST /host/ai/api/proposal_rejection_reasons/job_features_extraction
Content-Type: application/json
```

### Request Shape

```json
{
  "job_description": "string"
}
```

### Success Response

```json
{
  "success": true,
  "message": "message to system",
  "job_tools": ["string"],
  "job_requirements": ["string"],
  "job_key_points": ["string"]
}
```

### Failure Response

```json
{
  "success": false,
  "message": "message to system",
  "job_tools": null,
  "job_requirements": null,
  "job_key_points": null
}
```

---

## Task 2: Proposal Analysis

### Purpose

Analyzes a freelancer proposal against the job description and the previously extracted job features.

### Endpoint

```http
POST /host/ai/api/proposal_rejection_reasons/proposal_analysis
Content-Type: application/json
```

### Request Shape

```json
{
  "job_description": "string",
  "proposal_text": "string",
  "job_features": {}
}
```

### Success Response

```json
{
  "success": true,
  "message": "message to system",
  "final_report": "string"
}
```

### Failure Response

```json
{
  "success": false,
  "message": "message to system",
  "final_report": null
}
```

---

# Backend Integration Notes

## General Rules

* All JSON routes must use:

```http
Content-Type: application/json
```

* Image upload routes must use:

```http
Content-Type: multipart/form-data
```

* The backend should store reusable AI outputs whenever possible, especially:

  * Identity embeddings.
  * Freelancer embeddings.
  * Job embeddings.
  * Extracted job features.
  * Extracted profile features.
  * Proposal analysis reports.

* Reports that are expensive to generate should be created once and stored in the database, then retrieved when needed.

---

## Recommended Storage Responsibilities

| Feature                    | Output to Store            | Recommended Storage |
| -------------------------- | -------------------------- | ------------------- |
| Identity Recognition       | `person_embeddings`        | Vector database     |
| Job Recommendation System  | Freelancer embeddings      | Vector database     |
| Job Recommendation System  | Job embeddings             | Vector database     |
| Profile Analysis           | Extracted profile features | Main database       |
| Profile Analysis           | Final profile report       | Main database       |
| Proposal Rejection Reasons | Job features               | Main database       |
| Proposal Rejection Reasons | Proposal report            | Main database       |

---

# Setup

## 1. Install Python

Install Python 3.10 or later.

Check your installed version:

```bash
python --version
```

or:

```bash
python3 --version
```

---

## 2. Create and Activate a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 4. Create Environment File

```bash
cp .env.example .env
```

Then fill `.env` with the required environment variables.

---

## 5. Run the Application

```bash
uvicorn main:app --reload
```

The service should now be available locally.

Example:

```http
http://localhost:8000/host/ai/api
```

---

# Notes for Developers

* Keep request and response schemas synchronized with the Pydantic models in the `models/` directory.
* Any change in endpoint paths, field names, or response keys should be reflected in this README.
* AI models and agents should be called only from controllers, not directly from route files.
* Route files should remain thin and responsible mainly for request validation and response forwarding.
* Controllers should handle orchestration, logging, model calls, and error handling.
* Embedding outputs should not be recalculated unnecessarily if they are already stored and valid.
