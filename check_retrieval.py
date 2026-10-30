
from rag import load_vectorstore

db = load_vectorstore()

question = "Is the minimum attendance requirement 75% or 80%?"

results = db.similarity_search_with_score(question, k=3)

for i, (doc, score) in enumerate(results, start=1):
    print(f"\n--- Result {i} | Distance: {score:.4f} ---")
    print("Source:", doc.metadata.get("source"))
    print("Text:", doc.page_content)
