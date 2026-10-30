#This is the interface students will use. It displays answers, source files, PDF page numbers where available, and the retrieved evidence.


import streamlit as st

from rag import answer_question, load_vectorstore

st.set_page_config(
    page_title="College Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 College Knowledge Assistant")
st.write(
    "Ask questions about the college documents. "
    "Answers should be supported by the sources shown below."
)

@st.cache_resource
def get_database():
    return load_vectorstore()


try:
    with st.spinner("Loading college knowledge base..."):
        db = get_database()
    st.success("Knowledge base loaded.")
except Exception as exc:
    st.error(str(exc))
    st.info("First run `python ingest.py` in the project terminal.")
    st.stop()

question = st.text_input(
    "Enter your question",
    placeholder="What is the minimum attendance requirement?"
)

if st.button("Ask question", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Searching documents and preparing an answer..."):
            result = answer_question(question.strip(), db)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Supporting sources")

        if not result["sources"]:
            st.warning("No source passages were retrieved.")
        else:
            for i, source in enumerate(result["sources"], start=1):
                if source["page"] is not None:
                    label = f'{source["source"]} — page {source["page"]}'
                else:
                    label = source["source"]

                with st.expander(f"Source {i}: {label}"):
                    st.write(source["content"])

    except Exception as exc:
        st.error(f"Unable to answer the question: {exc}")
