
import json
import math
import re
from pathlib import Path
from collections import Counter

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "articles.json"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def load_articles() -> list[dict]:
    if not DATA_FILE.exists():
        return []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    return [
        article for article in articles
        if article.get("content") and article.get("url")
    ]


def search_articles(question: str, top_k: int = 3) -> list[dict]:
    articles = load_articles()
    query_terms = tokenize(question)

    if not articles or not query_terms:
        return []

    documents = [
        tokenize(article.get("title", "") + " " + article["content"])
        for article in articles
    ]

    document_frequency = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))

    average_length = sum(map(len, documents)) / len(documents)
    total_documents = len(documents)
    scores = []

    # BM25-style lexical ranking
    k1 = 1.5
    b = 0.75

    for article, tokens in zip(articles, documents):
        frequencies = Counter(tokens)
        document_length = len(tokens)
        score = 0.0

        for term in set(query_terms):
            term_frequency = frequencies[term]
            if term_frequency == 0:
                continue

            df = document_frequency[term]
            idf = math.log(
                1 + (total_documents - df + 0.5) / (df + 0.5)
            )

            denominator = term_frequency + k1 * (
                1 - b + b * document_length / max(average_length, 1)
            )

            score += idf * (
                term_frequency * (k1 + 1) / denominator
            )

        if score > 0:
            scores.append({
                "title": article.get("title", "Untitled"),
                "url": article["url"],
                "content": article["content"],
                "score": round(score, 4),
            })

    scores.sort(key=lambda item: item["score"], reverse=True)
    return scores[:top_k]
