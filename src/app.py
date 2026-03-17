import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.config_manager import get_config
from src.retriever import get_retriever
from src.tracing import traced_base_answer, traced_helper_answer, traced_strict_rag_answer
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Fleet Insights RAG", layout="wide")
st.title("RAG Research Assistant")

config = get_config()

st.sidebar.markdown("### Configuration")
st.sidebar.info(f"**Embedding Mode:** {config.embedding_mode}")

user_id = st.sidebar.text_input(
    "User ID",
    value="",
    placeholder="e.g. alice",
    help="Leave blank for unrestricted access. Roles are configured in roles.yaml.",
)

try:
    retriever = get_retriever(user_id=user_id or None)
    if user_id:
        from src.auth import get_user_roles
        roles = get_user_roles(user_id)
        st.sidebar.success(f"**Roles:** {', '.join(roles)}")
except RuntimeError:
    st.error(
        "No LanceDB index found. Ingest documents first:\n\n"
        "`python -m src.ingest`  or use the **Admin UI**."
    )
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

query = st.text_input("Ask a question about your documents:")

mode = st.radio(
    "Choose mode:",
    ["No-RAG", "Helper-RAG", "Strict-RAG"],
    index=1,
    horizontal=True,
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
            st.markdown(f"**{i}.** `{doc.metadata.get('source')}` | Page {doc.metadata.get('page')}")
            st.caption(doc.page_content[:300] + "...")
