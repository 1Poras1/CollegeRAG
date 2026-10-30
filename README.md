# College Knowledge Assistant (RAG)

## 1. Project Overview

The College Knowledge Assistant is a Retrieval-Augmented Generation (RAG) application that answers student questions using college documents.

Students can ask questions about admission, attendance, examinations, fees, and scholarships.

The application retrieves relevant information from the documents and uses a local language model to generate answers. It also displays supporting source documents so users can verify the information.

The assistant is designed to avoid inventing answers when the required information is not available.

## 2. Technologies Used

- Python
- Streamlit – User interface
- LangChain – RAG workflow
- FAISS – Vector database
- HuggingFace Sentence Transformers – Text embeddings
- Ollama – Local language model execution
- Gemma 2 2B – Answer generation
- PyPDF – PDF document loading

## 3. Project Structure

```text
CollegeRAG/
│
├── documents/
│   ├── admission_policy.txt
│   ├── attendance_policy.txt
│   ├── attendance_update.txt
│   ├── exam_rules.txt
│   ├── fee_policy.txt
│   └── scholarship_policy.txt
│
├── tests/
│   └── test_questions.json
│
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
│
├── app.py
├── ingest.py
├── rag.py
├── check_retrieval.py
├── requirements.txt
└── README.md
```

## 4. System Architecture

The project follows a Retrieval-Augmented Generation architecture.

1. Document Loading: The ingestion module reads PDF and TXT files from the documents folder.
2. Text Splitting: Documents are divided into smaller text chunks.
3. Embedding Generation: Sentence Transformers converts text chunks into numerical vectors.
4. Vector Storage: FAISS stores the vectors for efficient similarity search.
5. Retrieval: When a student asks a question, relevant passages are retrieved from FAISS.
6. Answer Generation: Gemma 2 2B generates an answer using the retrieved context.
7. Source Display: Streamlit displays the answer and supporting document passages.

## 5. Installation and Setup

### Prerequisites

- Python 3.10 or later
- Ollama installed
- VS Code or another Python editor

### Step 1: Create a virtual environment

```powershell
python -m venv myenv
```

### Step 2: Activate the environment

```powershell
.\myenv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Download the language model

```powershell
ollama pull gemma2:2b
```

### Step 5: Build the knowledge base

```powershell
python ingest.py
```

### Step 6: Start the application

```powershell
python -m streamlit run app.py
```

Open the local Streamlit address displayed in the terminal.

## 6. Running the Retrieval Check

To inspect retrieved passages, run:

```powershell
python check_retrieval.py
```

This displays the retrieved document passages and their similarity distances.

## 7. Special Handling of Conflicting Information

The sample knowledge base contains two different attendance requirements:

- attendance_policy.txt specifies 75%.
- attendance_update.txt specifies 80%.

The application includes a dedicated conflict-handling rule for this sample attendance policy. It reports both values