import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # load .env for OPENAI_API_KEY, PYTHONPATH etc.

from src.embed_store import load_faiss
from src.config_manager import get_config
from src.tracing import traced_base_answer, traced_helper_answer, traced_strict_rag_answer
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Fleet Insights RAG", layout="wide")
st.title("RAG Research Assistant")

# Load configuration
config = get_config()

# Show embedding mode
st.sidebar.markdown("### Configuration")
st.sidebar.info(f"**Embedding Mode:** {config.embedding_mode}")
st.sidebar.info(f"**Incremental:** {'Enabled' if config.is_incremental_enabled() else 'Disabled'}")

# Load FAISS retriever
db = load_faiss()
if db is None:
    st.error(
        "No FAISS index found. Please ingest documents first by running:\n\n"
        "`python -m src.ingest`"
    )
    st.stop()
retriever = db.as_retriever(search_kwargs={"k": config.get("top_k", 3)})

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
query = st.text_input("Ask a question about your documents:")

mode = st.radio(
    "Choose mode:",
    ["No-RAG", "Helper-RAG", "Strict-RAG"],
    index=1,  # default Helper
    horizontal=True
)

if query:
    with st.spinner("Thinking..."):
        if mode == "No-RAG":
            result = traced_base_answer(query, llm)
        elif mode == "Helper-RAG":
            result = traced_helper_answer(query, retriever, llm)
        else:
            result = traced_strict_rag_answer(query, retriever, llm)

    st.subheader("Answer")
    st.write(result["answer"])

    if result.get("augment"):
        st.subheader("Additional Context")
        st.write(result["augment"])

    if result["sources"]:
        st.subheader("Sources")
        for i, doc in enumerate(result["sources"], 1):
            st.markdown(f"**{i}.** File: `{doc.metadata.get('source')}` | Page: {doc.metadata.get('page')}")
            st.caption(doc.page_content[:300] + "...")
