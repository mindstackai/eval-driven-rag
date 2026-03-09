"""
Answer quality evaluation using an LLM judge.
"""


def faithfulness_score(answer: str, context_chunks: list[str], llm_client) -> float:
    """
    Use LLM to score whether the answer is grounded in the context chunks.
    Prompt the LLM to return a score 0.0-1.0.

    A score of 1.0 means every claim in the answer is directly supported by
    the provided context.  A score of 0.0 means the answer contains no
    information that can be traced back to the context.

    Args:
        answer: The generated answer to evaluate.
        context_chunks: List of retrieved text chunks used to produce the answer.
        llm_client: An object with an ``invoke(prompt: str) -> object`` method
                    whose return value has a ``.content`` attribute (compatible
                    with LangChain ChatOpenAI).

    Returns:
        Faithfulness score as a float in [0.0, 1.0].
    """
    context_text = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = f"""You are an evaluation assistant. Your task is to judge whether an answer
is faithfully grounded in the provided context chunks.

Context:
{context_text}

Answer:
{answer}

Instructions:
- Score 1.0 if every factual claim in the answer is directly supported by the context.
- Score 0.0 if the answer contains fabricated or unsupported claims.
- Use intermediate values (e.g. 0.5, 0.7) for partial grounding.
- Respond with ONLY a single decimal number between 0.0 and 1.0, nothing else.

Score:"""

    response = llm_client.invoke(prompt)
    raw = response.content.strip()
    try:
        score = float(raw)
    except ValueError:
        # Extract first float-like token from the response as a fallback
        import re
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", raw)
        score = float(match.group()) if match else 0.0
    return max(0.0, min(1.0, score))


def answer_relevance_score(question: str, answer: str, llm_client) -> float:
    """
    Use LLM to score whether the answer addresses the question.
    Prompt the LLM to return a score 0.0-1.0.

    A score of 1.0 means the answer completely and directly addresses the
    question.  A score of 0.0 means the answer is entirely off-topic.

    Args:
        question: The original question posed to the RAG system.
        answer: The generated answer to evaluate.
        llm_client: An object with an ``invoke(prompt: str) -> object`` method
                    whose return value has a ``.content`` attribute (compatible
                    with LangChain ChatOpenAI).

    Returns:
        Answer relevance score as a float in [0.0, 1.0].
    """
    prompt = f"""You are an evaluation assistant. Your task is to judge how well an answer
addresses the given question.

Question:
{question}

Answer:
{answer}

Instructions:
- Score 1.0 if the answer fully and directly addresses the question.
- Score 0.0 if the answer is completely off-topic or does not address the question at all.
- Use intermediate values (e.g. 0.5, 0.7) for partial relevance.
- Respond with ONLY a single decimal number between 0.0 and 1.0, nothing else.

Score:"""

    response = llm_client.invoke(prompt)
    raw = response.content.strip()
    try:
        score = float(raw)
    except ValueError:
        import re
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", raw)
        score = float(match.group()) if match else 0.0
    return max(0.0, min(1.0, score))
