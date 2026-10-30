#This stage reads every supported PDF and TXT file, splits its text into chunks, generates embeddings, and saves the FAISS database.


from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


def load_documents():
    documents = []

    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if path.suffix.lower() == ".txt":
            loader = TextLoader(
                str(path),
                autodetect_encoding=True
            )
            loaded = loader.load()

        elif path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            loaded = loader.load()

        else:
            continue

        # Preserve the filename and PDF page metadata.
        for doc in loaded:
            doc.metadata["source"] = path.name

        documents.extend(loaded)

    if not documents:
        raise ValueError(
            "No PDF or TXT documents found in the documents folder."
        )

    return documents


def build_vectorstore():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        raise ValueError("The documents contain no extractable text.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(VECTORSTORE_DIR))

    print(f"Documents loaded: {len(documents)}")
    print(f"Text chunks created: {len(chunks)}")
    print(f"FAISS index saved to: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()
