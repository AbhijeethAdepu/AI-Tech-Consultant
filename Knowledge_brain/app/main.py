
from fastapi import FastAPI
from pydantic import BaseModel, Field
import httpx

from Knowledge_brain.app.semantic_retrieval import search_semantic

app = FastAPI(
    title="Knowledge Brain API",
    description="Documentation-grounded AI support service",
    version="0.3.0",
)

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2:3b"


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Source(BaseModel):
    title: str
    url: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    status: str
    sources: list[Source] = Field(default_factory=list)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "knowledge-brain",
    }


def generate_answer(question: str, results: list[dict]) -> str:
    context = "\n\n".join(
        f"Source: {item['title']}\n"
        f"URL: {item['url']}\n"
        f"Documentation:\n{item['content'][:2500]}"
        for item in results
    )

    prompt = f"""
You are a technical support assistant for Brilliant Directories.

Answer the user's question using ONLY the supplied documentation.
Do not invent settings, features, URLs, or instructions.
If the documentation does not contain enough evidence, say:
"I couldn't verify this from the available documentation. Please escalate to a human support specialist."

Give concise, practical instructions when the evidence supports them.
Do not claim that you performed any action.

DOCUMENTATION:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=180.0,
    )
    response.raise_for_status()

    answer = response.json().get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    return answer



@app.post("/query", response_model=QueryResponse)
def query_knowledge(request: QueryRequest):
    results = search_semantic(request.question, top_k=3)

    if not results:
        return QueryResponse(
            answer=(
                "I couldn't find relevant information in the available "
                "documentation. Please escalate to a human support specialist."
            ),
            status="insufficient_evidence",
        )

    sources = [
        Source(
            title=item["title"],
            url=item["url"],
            relevance_score=round(item["score"], 4),
        )
        for item in results
    ]

    # Initial threshold only; calibrate with a larger test set.
    MIN_RELEVANCE_SCORE = 0.55

    if results[0]["score"] < MIN_RELEVANCE_SCORE:
        return QueryResponse(
            answer=(
                "I couldn't find sufficiently relevant documentation to "
                "answer this question reliably. Please escalate to a human "
                "support specialist."
            ),
            status="insufficient_evidence",
            sources=sources,
        )

    try:
        answer = generate_answer(request.question, results)
    except (httpx.HTTPError, RuntimeError) as exc:
        return QueryResponse(
            answer=(
                f"Answer generation is temporarily unavailable "
                f"({type(exc).__name__}). Please retry or escalate."
            ),
            status="generation_error",
            sources=sources,
        )

    # Detect the explicit fallback requested in the generation prompt.
    if "couldn't verify this from the available documentation" in answer.lower():
        return QueryResponse(
            answer=answer,
            status="insufficient_evidence",
            sources=sources,
        )

    return QueryResponse(
        answer=answer,
        status="answered",
        sources=sources,
    )
