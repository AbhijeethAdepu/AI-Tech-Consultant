
import json
import math
from pathlib import Path

import httpx

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent
ARTICLES_FILE = KNOWLEDGE_DIR / "data" / "articles.json"
EMBEDDINGS_FILE = KNOWLEDGE_DIR / "data" / "embeddings.json"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"


def get_embedding(text: str) -> list[float]:
    response = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60.0,
    )
    response.raise_for_status()
    embeddings = response.json().get("embeddings", [])

    if not embeddings:
        raise RuntimeError("Ollama returned no embeddings.")

    return embeddings[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def load_articles() -> list[dict]:
    if not ARTICLES_FILE.exists():
        raise FileNotFoundError(f"Articles file not found: {ARTICLES_FILE}")

    with ARTICLES_FILE.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    return [
        article for article in articles
        if article.get("title") and article.get("content")
    ]


def article_text(article: dict) -> str:
    return (
        f"Title: {article.get('title', '')}\n"
        f"Content: {article.get('content', '')[:6000]}"
    )


def load_or_create_embeddings(articles: list[dict]) -> list[list[float]]:
    fingerprint = [
        {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "content": article.get("content", ""),
        }
        for article in articles
    ]

    if EMBEDDINGS_FILE.exists():
        try:
            with EMBEDDINGS_FILE.open("r", encoding="utf-8") as file:
                cache = json.load(file)

            if cache.get("articles") == fingerprint:
                vectors = cache.get("embeddings", [])
                if len(vectors) == len(articles):
                    print("Using cached article embeddings.")
                    return vectors
        except (json.JSONDecodeError, OSError):
            print("Cache unreadable; regenerating embeddings.")

    print(f"Creating embeddings for {len(articles)} articles...")

    vectors = []
    for index, article in enumerate(articles, start=1):
        print(f"Embedding article {index}/{len(articles)}...")
        vectors.append(get_embedding(article_text(article)))

    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with EMBEDDINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            {"articles": fingerprint, "embeddings": vectors},
            file,
        )

    print(f"Saved embedding cache: {EMBEDDINGS_FILE}")
    return vectors


def search_semantic(question: str, top_k: int = 3) -> list[dict]:
    articles = load_articles()

    if not articles:
        return []

    article_embeddings = load_or_create_embeddings(articles)
    question_embedding = get_embedding(question)

    results = []

    for article, embedding in zip(articles, article_embeddings):
        results.append({
            "title": article.get("title", "Untitled"),
            "url": article.get("url", ""),
            "content": article.get("content", ""),
            "score": cosine_similarity(question_embedding, embedding),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]
