# Go Freelance - AI Features

This repository contains the implementation of the AI features for the GoFreelance platform. Each feature has its own sub-directory, dependencies, documentation, and implementation.

These features aim to achieve the platform's core goal which is connecting clients with freelancers professionally helping startup freelancers land jobs faster and helping clients find the right candidates for their jobs.

The main AI features employed to achieve this are:

- AI Profile Analyzer & Enhancer
- AI Recommendation System
- Job Description Enhancement
- Proposal Rejection Reasons

In addition to that, it is necessary to prevent harmfulness, cheating, scams, hacking, or any disrespectful behaviors on the platform.

The feature employed for this purpose is:

- Identity Recognition



## AI Features

The implemented AI features are described as follows:

### AI Profile Analyzer & Enhancer

A freelancer's portfolio is one of the most important factors affecting their visibility on the platform, so it is crucial to build a strong, professional, and era-adapted portfolio.

This feature aims to analyze the freelancer's portfolio on our platform, highlight strengths and weaknesses, and provide suggestions for enhancement. This helps freelancers address visibility issues and the always-being-refused dilemma.

---

### Recommendation System

Often, clients struggle to find the appropriate freelancers for their requirements, and freelancers also struggle to reach jobs that match their skills and abilities.

The AI Recommendation System solves these problems by recommending the best suitable jobs for freelancers. The recommendation process is mainly based on three aspects:

#### 1. Freelancer Data — Content-Based Filtering

Here, the freelancer's job title, portfolio, skills, and previous work on the platform are considered. The relevance between this data and the job posted by the client is measured to decide whether the job is a good match for the freelancer or not.

#### 2. Similar Freelancers Approach — Collaborative Filtering

Here, the freelancer is recommended jobs that similar freelancers in terms of portfolio, job title, skills, and other data usually apply to. This ensures that the jobs recommended to the freelancer follow the most recent updates and trends in their field.

#### 3. Basic Filtering

After the AI recommends the best matches for a freelancer, the platform still provides a backend metadata filtering mechanism that allows freelancers to choose whatever they want.

This ensures that freelancers have full flexibility in exploring other job opportunities, price ranges, categories, and other available options.

---

### Identity Recognition

In a platform that contains financial data and users' private data, it is crucial to ensure that all users follow respectful and professional behaviors while dealing with others on the platform.

Therefore, a strict ban system is employed to provide only one account for each user. Hence, if a user does not follow the professional manners of the platform, they get banned and cannot sign up again.

The technology used behind this feature is Face Recognition technology, which is commonly used for security purposes such as in our case.

The methodology is done as follows:

1. When a user signs up for the first time, they are asked to upload two images: one for their own face and another for a national identity card. The system ensures that the person has a reliable national identity.

2. The facial features of the person are then saved as high-dimensional embeddings, also known as vectors, in a vector database using PostgreSQL.

3. If another user tries to sign up later, their facial embeddings are compared with our recorded embeddings in the vector database. If a match is found, the sign-up process is refused.

This way, the banned user cannot re-sign up.

In order to reach the best possible result, various AI models are explored, trained, and evaluated. Finally, the champion model is deployed as the main Face Recognition model in our platform.

---

### Job Description Enhancement

---

### Proposal Rejection Reasons


## Requirements
- Python 3.10 or later.
- Supported LLM provider API Keys.
