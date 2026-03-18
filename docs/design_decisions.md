# Design Decisions

## Chunking Strategy

## Embedding Model Selection

The project supports four embedding modes, controlled by `EMBEDDING_MODE` in `.env` (or `embedding.mode` in `config.yaml`).

| Mode | Model | Dims | Cost | Notes |
|---|---|---|---|---|
| `openai` | `text-embedding-3-small` | 1536 | Paid | Recommended for production |
| `openai-legacy` | `text-embedding-ada-002` | 1536 | Paid | Legacy; prefer `openai` |
| `local` | `all-MiniLM-L6-v2` | 384 | Free | Sentence-transformers; good for local dev / CI |

**Auto-selection order** (`EMBEDDING_MODE=auto`):
1. `openai` — if `OPENAI_API_KEY` is set
2. `local` — fallback

---

## Vector Store: FAISS vs LanceDB

## Eval Framework

## Access Control (Future)

### Overview

The project plans to move from FAISS to **LanceDB** as the primary vector store. LanceDB is file-based (or S3/GCS-backed) and stores metadata alongside each vector row, making it natural to enforce **document-level access control** at query time.

### Design: Role-based chunk filtering

Each chunk is stored with an `allowed_roles` metadata field (e.g. `"analyst"`, `"admin"`, `"public"`). At query time the caller's role is passed as a LanceDB `WHERE` filter, so only authorised chunks are ranked and returned.

```
chunk row = { vector, text, source, page, chunk_id, allowed_roles }
```

The `LanceDBStore.similarity_search()` already accepts a `filters` parameter that translates to a SQL-style `WHERE` clause — no schema change is needed when the feature is activated.

```python
# Ingest — tag chunks with their permitted role
store.add_documents(
    chunks=texts,
    embeddings=vectors,
    metadatas=[{"source": "report.pdf", "allowed_roles": "analyst"}, ...],
)

# Query — only return chunks the user's role can see
results = store.similarity_search(
    query_embedding=q_vec,
    k=5,
    filters={"allowed_roles": user_role},
)
```

### LanceDB access control in production (planned)

When LanceDB is the primary store, row-level security can be layered as follows:

1. **Ingest pipeline** — the document owner / classifier assigns `allowed_roles` at chunk creation time (e.g. derived from GCS object ACLs or a document-level permissions table).
2. **Retriever** — `LanceDBStore.similarity_search(filters={"allowed_roles": session_role})` is called with the authenticated user's role.
3. **Auth middleware** — the calling service validates the user's identity (e.g. via Google IAP or a JWT) and maps it to a role before invoking the retriever. The retriever itself is stateless and trusts the caller to pass the correct role.

This keeps access control enforcement at a single choke-point (the retriever call) rather than scattered across the application.
