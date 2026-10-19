

\# AI Tech Consultant



An AI-powered support automation project consisting of two Python services.



\## Components



\### 1. Knowledge Brain

\- Ingests support documentation.

\- Uses Ollama embeddings for semantic retrieval.

\- Generates documentation-grounded answers.

\- Returns source URLs and relevance scores.

\- Flags questions when available evidence is insufficient.



\### 2. Inbox Agent

\- Classifies incoming customer emails.

\- Requests relevant information from the Knowledge Brain.

\- Generates customer-support reply drafts.

\- Stores drafts in SQLite.

\- Supports human approval, editing, and rejection.



\## Technology Stack



\- Python

\- FastAPI

\- Ollama

\- `nomic-embed-text`

\- `llama3.2:3b`

\- SQLite

\- HTTPX



\## Prerequisites



\- Python

\- Ollama

\- The `nomic-embed-text` and `llama3.2:3b` models



Install the models:



```bash

ollama pull nomic-embed-text

ollama pull llama3.2:3b

```



Install Python dependencies:



```bash

pip install -r requirements.txt

```



\## Run the services



From the project root, start the Knowledge Brain:



```bash

python -m uvicorn Knowledge\_brain.app.main:app --reload --port 8000

```



Start the Inbox Agent in a second terminal:



```bash

python -m uvicorn inbox\_agent.app.main:app --reload --port 8001

```



\## API Documentation



\- Knowledge Brain: http://127.0.0.1:8000/docs

\- Inbox Agent: http://127.0.0.1:8001/docs



\## Current Limitations



This is an in-progress MVP. Documentation coverage, retrieval quality, evaluation, and the complete end-to-end workflow still require validation. Email sending is not implemented; generated replies are saved for human review.



