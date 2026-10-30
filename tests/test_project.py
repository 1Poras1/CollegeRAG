import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
QUESTIONS_FILE = Path(__file__).parent / "test_questions.json"


def test_required_documents_exist():
    required_files = [
        "admission_policy.txt",
        "attendance_policy.txt",
        "attendance_update.txt",
        "exam_rules.txt",
        "fee_policy.txt",
        "scholarship_policy.txt",
    ]

    for filename in required_files:
        assert (DOCUMENTS_DIR / filename).exists()


def test_at_least_five_documents():
    documents = list(DOCUMENTS_DIR.glob("*.txt")) + list(
        DOCUMENTS_DIR.glob("*.pdf")
    )
    assert len(documents) >= 5


def test_question_bank_has_ten_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    assert len(questions) >= 10


def test_all_questions_have_required_fields():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    for question in questions:
        assert "id" in question
        assert "question" in question
        assert "expected" in question

def test_pdf_document_is_loaded():
    from rag import load_vectorstore

    db = load_vectorstore()
    results = db.similarity_search(
        "How many books can a student borrow at one time?",
        k=3
    )

    combined_text = " ".join(doc.page_content for doc in results)

    assert "three books" in combined_text.lower()

def test_pdf_question_answer():
    from rag import answer_question

    result = answer_question(
        "How many books can a student borrow at one time?"
    )

    assert "three books" in result["answer"].lower()
    assert any(
    source["source"] == "sample_college_library_policy.pdf"
    for source in result["sources"]
)       

def test_unanswerable_question():
    from rag import answer_question

    result = answer_question(
        "What is the maximum number of days a student can borrow a laptop?"
    )

    assert (
        "could not find enough information"
        in result["answer"].lower()
    )

def test_conflicting_attendance_policies():
    from rag import answer_question

    result = answer_question(
        "Is the minimum attendance requirement 75% or 80%?"
    )

    answer = result["answer"].lower()

    assert "75%" in answer
    assert "80%" in answer
    assert "conflicting" in answer    