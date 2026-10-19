from Knowledge_brain.app.semantic_retrieval import search_semantic

print("Starting semantic search test...")

question = "How do I configure my website?"
results = search_semantic(question)

print(f"Results returned: {len(results)}")

for index, result in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print("Title:", result["title"])
    print("Score:", round(result["score"], 4))
    print("URL:", result["url"])
    print("Content:", result["content"][:300])