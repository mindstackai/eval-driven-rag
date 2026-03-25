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

# ── Sidebar: identity & access ─────────────────────────────────────────────
st.sidebar.markdown("### Identity")
user_id = st.sidebar.text_input(
    "User ID",
    value="",
    placeholder="e.g. alice, bob, carol",
    help="Leave blank for unrestricted access. Roles are configured in roles.yaml.",
)

from src.auth import get_user_role, get_user_roles, list_users

if user_id:
    assigned_role = get_user_role(user_id)
    expanded_roles = get_user_roles(user_id)

    role_colour = {"admin": "🔴", "analyst": "🟡", "public": "🟢"}.get(assigned_role, "⚪")
    st.sidebar.markdown(
        f"**{role_colour} {user_id}** — `{assigned_role}`\n\n"
        f"Can read: {' · '.join(f'`{r}`' for r in expanded_roles)}"
    )
else:
    st.sidebar.info("No user — unrestricted access (all chunks visible).")

st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration")
st.sidebar.info(f"**Embedding:** {config.embedding_mode}")

# ── Load retriever ─────────────────────────────────────────────────────────
try:
    retriever = get_retriever(user_id=user_id or None)
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
            role = doc.metadata.get("allowed_roles", "?")
            role_badge = {"admin": "🔴 admin", "analyst": "🟡 analyst", "public": "🟢 public"}.get(role, f"⚪ {role}")
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**{i}.** `{source}` | Page {page} | {role_badge}")
            st.caption(doc.page_content[:300] + "...")
