"""
Admin UI — upload documents, assign access roles, manage users.

Run from the eval-driven-rag/ directory:
    streamlit run src/admin/app.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on the path when running via `streamlit run`
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.auth import VALID_ROLES, get_user_roles, list_users, remove_user, set_user_role
from src.config_manager import get_config
from src.ingest import ingest_file_to_lancedb
from src.vectorstore.lancedb_store import LanceDBStore

st.set_page_config(page_title="RAG Admin", page_icon="🔐", layout="wide")
st.title("🔐 RAG Admin Panel")

config = get_config()

upload_tab, users_tab, index_tab = st.tabs(["Upload & Ingest", "Manage Users", "Index Overview"])

# ──────────────────────────────────────────────
# Tab 1 — Upload & Ingest
# ──────────────────────────────────────────────
with upload_tab:
    st.subheader("Upload Document to LanceDB")
    st.caption(
        "Files are chunked, embedded, and stored in LanceDB with the selected access role. "
        "The existing FAISS index is not affected."
    )

    uploaded = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])

    col1, col2 = st.columns([1, 2])
    with col1:
        role = st.selectbox(
            "Access role",
            VALID_ROLES,
            help="Chunks will only be returned to users whose role includes this level.",
        )
    with col2:
        st.markdown("")
        st.markdown("")
        role_preview = {
            "public":  "Readable by everyone (public, analyst, admin)",
            "analyst": "Readable by analyst and admin only",
            "admin":   "Readable by admin only",
        }
        st.info(role_preview[role])

    if uploaded:
        st.write(f"**File:** {uploaded.name}  |  **Size:** {len(uploaded.getvalue()) / 1024:.1f} KB")

        if st.button("Ingest into LanceDB", type="primary"):
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            try:
                with st.spinner(f"Chunking and embedding {uploaded.name}..."):
                    n = ingest_file_to_lancedb(tmp_path, role, config)
                st.success(f"Ingested **{n} chunks** from `{uploaded.name}` with role `{role}`.")
            except Exception as e:
                st.error(f"Ingest failed: {e}")
            finally:
                tmp_path.unlink(missing_ok=True)

# ──────────────────────────────────────────────
# Tab 2 — Manage Users
# ──────────────────────────────────────────────
with users_tab:
    st.subheader("Users & Roles")

    users = list_users()

    if users:
        rows = []
        for uid, r in users.items():
            rows.append({
                "User ID": uid,
                "Assigned Role": r,
                "Can Access": ", ".join(get_user_roles(uid)),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No users configured yet. Add one below.")

    st.divider()

    col_add, col_remove = st.columns(2)

    with col_add:
        st.markdown("**Add / Update User**")
        new_uid = st.text_input("User ID", placeholder="e.g. john")
        new_role = st.selectbox("Role", VALID_ROLES, key="add_role")
        if st.button("Save User"):
            if new_uid.strip():
                set_user_role(new_uid.strip(), new_role)
                st.success(f"Saved `{new_uid}` → `{new_role}`")
                st.rerun()
            else:
                st.warning("Enter a user ID.")

    with col_remove:
        st.markdown("**Remove User**")
        if users:
            del_uid = st.selectbox("Select user to remove", list(users.keys()))
            if st.button("Remove User", type="secondary"):
                remove_user(del_uid)
                st.success(f"Removed `{del_uid}`.")
                st.rerun()
        else:
            st.caption("No users to remove.")

# ──────────────────────────────────────────────
# Tab 3 — Index Overview
# ──────────────────────────────────────────────
with index_tab:
    st.subheader("LanceDB Index Overview")

    try:
        store = LanceDBStore(
            db_path=config.get_lancedb_path(),
            table_name=config.get_lancedb_table(),
        )
        store.load()
        tbl = store._open_table()
        df = tbl.to_pandas()

        # ── Metrics row ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Chunks", f"{len(df):,}")
        c2.metric(
            "Unique Sources",
            df["source"].nunique() if "source" in df.columns else "—",
        )
        c3.metric(
            "Access Roles in Use",
            df["allowed_roles"].nunique() if "allowed_roles" in df.columns else "—",
        )
        vec_col = "vector"
        dims = len(df[vec_col].iloc[0]) if vec_col in df.columns and len(df) > 0 else "—"
        c4.metric("Embedding Dims", dims)

        st.divider()

        # ── Role distribution chart ──
        if "allowed_roles" in df.columns:
            st.markdown("**Chunks per role**")
            role_counts = df["allowed_roles"].value_counts().reset_index()
            role_counts.columns = ["Role", "Chunks"]
            st.bar_chart(role_counts.set_index("Role"))

        # ── Per-source breakdown ──
        if "source" in df.columns and "allowed_roles" in df.columns:
            st.markdown("**Documents in index**")
            summary = (
                df.groupby(["source", "allowed_roles"])
                .size()
                .reset_index(name="chunks")
                .rename(columns={"source": "Source", "allowed_roles": "Role", "chunks": "Chunks"})
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Raw chunk preview ──
        with st.expander("Raw chunk preview (first 50 rows)"):
            display_cols = [c for c in df.columns if c != "vector"]
            st.dataframe(df[display_cols].head(50), use_container_width=True)

    except RuntimeError:
        st.info(
            "No LanceDB index found yet. "
            "Go to **Upload & Ingest** to add your first document."
        )
    except Exception as e:
        st.error(f"Could not load index: {e}")
