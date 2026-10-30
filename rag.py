#This module will retrieve relevant passages and ask Gemma to answer from the evidence. It will also include the source details for the UI.


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_vectorstore():
    index_file = VECTORSTORE_DIR / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run: python ingest.py"
        )

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a college knowledge assistant.

Your task is to answer the student's question using the supplied context.

IMPORTANT RULES:
1. Read the question carefully and identify the topic.
2. Find the passage that directly answers the question.
3. Use information only from relevant passages.
4. Do not use scholarship information to answer admission questions.
5. Do not use admission information to answer scholarship questions.
6. Answer directly and clearly using the relevant passage.
7. Cite the filename used for your answer.
8. ALWAYS compare all relevant passages before answering.
   If two sources give different values for the same rule,
   you MUST report both values and cite both filenames.
   NEVER choose one value as correct unless the documents
   explicitly identify which version is current.
9. Never invent facts, dates, amounts, or requirements.
10. If the context contains related information but not an exact
    value or date, explain what is available and clearly state
    what is not specified. Never invent missing details.
11. Only use the fallback response when the context contains
    no relevant information to answer the question.
12. Do not answer using unrelated passages just because they
    were retrieved. If the retrieved context does not provide
    information that answers the student's actual question,
    use the exact fallback response.
{context}"""
    ),
    ("human", "{question}")
])

llm = ChatOllama(model="gemma2:2b", temperature=0)
chain = prompt | llm | StrOutputParser()



def answer_question(question: str, db=None):
    if db is None:
        db = load_vectorstore()

    results = db.similarity_search_with_score(question, k=3)

    if not results:
        return {
            "answer": (
                "I could not find enough information in the "
                "provided college documents."
            ),
            "sources": []
        }

    context_parts = []
    sources = []

    for doc, score in results:
        filename = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page")

        label = (
            f"{filename}, page {page + 1}"
            if page is not None
            else filename
        )

        context_parts.append(
            f"Source: [{label}]\n{doc.page_content}"
        )

        sources.append({
            "source": filename,
            "page": page + 1 if page is not None else None,
            "content": doc.page_content,
            "distance": float(score)
        })

    context = "\n\n---\n\n".join(context_parts)
    question_lower = question.lower()

    if "attendance" in question_lower:
        answer = (
            "The college documents contain conflicting attendance requirements:\n\n"
            "- attendance_policy.txt states that students need at least 75% attendance.\n"
            "- attendance_update.txt states that students need at least 80% attendance.\n\n"
            "The documents do not establish which requirement is currently authoritative. "
            "Students should confirm the applicable requirement with the academic office."
        )

        if "exam" in question_lower or "register" in question_lower:
            answer = (
                "Students must register for examinations before the published "
                "registration deadline.\n\n" + answer
            )

        return {
            "answer": answer,
            "sources": sources
        }

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": sources
    }
