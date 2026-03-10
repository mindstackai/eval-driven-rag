"""Thin tracing layer using EvalTrace's SpanRecorder.

Provides traced wrapper functions for each RAG mode.
Span attributes follow EvalTrace conventions so judge, latency,
and cost modules can extract data automatically.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from spanrecorder import SpanRecorder

# Module-level singleton
recorder = SpanRecorder()


def traced_base_answer(query: str, llm: ChatOpenAI) -> dict:
    """No-RAG mode with tracing."""
    with recorder.start_span("request", attrs={
        "component": "app",
        "user.query": query,
    }) as root:
        with recorder.start_span("llm.generation", attrs={"component": "llm"}) as llm_span:
            response = llm.invoke(f"Answer concisely:\n\nQ: {query}\nA:")
            answer = response.content
            usage = response.response_metadata.get("token_usage", {})
            llm_span.set_attribute("llm.model", llm.model_name)
            llm_span.set_attribute("llm.prompt_tokens", usage.get("prompt_tokens", 0))
            llm_span.set_attribute("llm.completion_tokens", usage.get("completion_tokens", 0))

        root.set_attribute("assistant.answer", answer)

    return {"answer": answer, "augment": None, "sources": []}


def traced_helper_answer(query: str, retriever, llm: ChatOpenAI) -> dict:
    """Helper-RAG mode with tracing."""
    with recorder.start_span("request", attrs={
        "component": "app",
        "user.query": query,
    }) as root:
        # Base LLM call
        with recorder.start_span("llm.base", attrs={"component": "llm"}) as base_span:
            base_response = llm.invoke(f"Answer concisely:\n\nQ: {query}\nA:")
            base_answer = base_response.content
            usage = base_response.response_metadata.get("token_usage", {})
            base_span.set_attribute("llm.model", llm.model_name)
            base_span.set_attribute("llm.prompt_tokens", usage.get("prompt_tokens", 0))
            base_span.set_attribute("llm.completion_tokens", usage.get("completion_tokens", 0))

        # Retrieval
        with recorder.start_span("retrieval.search", attrs={"component": "retriever"}) as ret_span:
            docs: List[Document] = retriever.invoke(query)
            usable_docs = [d for d in docs if len(d.page_content.strip()) > 50]
            passages = [d.page_content for d in usable_docs]
            ret_span.set_attribute("retrieval.passages", passages)

        if not usable_docs:
            root.set_attribute("assistant.answer", base_answer)
            return {"answer": base_answer, "augment": "No additional context.", "sources": []}

        # Augmentation LLM call
        context_blob = "\n\n".join(
            [f"[{i+1}] {d.page_content}" for i, d in enumerate(usable_docs)]
        )
        aug_prompt = (
            f"You already answered this question:\n\n"
            f"Question: {query}\nBase answer: {base_answer}\n\n"
            f"Now you are given optional context (may be redundant). "
            f"Only add details that are consistent and helpful.\n"
            f"If nothing adds value, return 'No additional context.'\n\n"
            f"Context:\n{context_blob}"
        )

        with recorder.start_span("llm.augment", attrs={"component": "llm"}) as aug_span:
            aug_response = llm.invoke(aug_prompt)
            aug_answer = aug_response.content
            usage = aug_response.response_metadata.get("token_usage", {})
            aug_span.set_attribute("llm.model", llm.model_name)
            aug_span.set_attribute("llm.prompt_tokens", usage.get("prompt_tokens", 0))
            aug_span.set_attribute("llm.completion_tokens", usage.get("completion_tokens", 0))

        root.set_attribute("assistant.answer", base_answer)

    return {"answer": base_answer, "augment": aug_answer, "sources": usable_docs}


def traced_strict_rag_answer(query: str, retriever, llm: ChatOpenAI) -> dict:
    """Strict-RAG mode with tracing."""
    with recorder.start_span("request", attrs={
        "component": "app",
        "user.query": query,
    }) as root:
        # Retrieval
        with recorder.start_span("retrieval.search", attrs={"component": "retriever"}) as ret_span:
            docs: List[Document] = retriever.invoke(query)
            passages = [d.page_content for d in docs]
            ret_span.set_attribute("retrieval.passages", passages)

        # LLM generation
        context = "\n\n".join(d.page_content for d in docs)
        prompt = (
            "Use only the following context to answer the question. "
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

        with recorder.start_span("llm.generation", attrs={"component": "llm"}) as llm_span:
            response = llm.invoke(prompt)
            answer = response.content
            usage = response.response_metadata.get("token_usage", {})
            llm_span.set_attribute("llm.model", llm.model_name)
            llm_span.set_attribute("llm.prompt_tokens", usage.get("prompt_tokens", 0))
            llm_span.set_attribute("llm.completion_tokens", usage.get("completion_tokens", 0))

        root.set_attribute("assistant.answer", answer)

    return {"answer": answer, "augment": None, "sources": docs}
