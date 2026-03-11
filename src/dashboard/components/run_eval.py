"""Run New Eval section — Phase 1 and Phase 2 forms."""
import json
import os
from datetime import datetime

import streamlit as st


def render_run_eval(ret_df_raw, all_strategies, results_dir, config_path):
    """Render the Run New Eval section with Phase 1 and Phase 2 forms.

    Args:
        ret_df_raw: Unfiltered retrieval DataFrame (for Phase 2 config preview).
        all_strategies: List of all available strategy names.
        results_dir: Path to eval results directory.
        config_path: Path to config.yaml.
    """
    st.header("Run New Eval")

    _render_phase1_form(results_dir, config_path)
    st.divider()
    _render_phase2_form(ret_df_raw, all_strategies, results_dir, config_path)


def _render_phase1_form(results_dir, config_path):
    """Render Phase 1 retrieval eval form."""
    with st.form("run_eval_form"):
        st.subheader("Run Phase 1: Retrieval Eval")
        st.markdown("Build indexes with different chunk sizes and evaluate retrieval quality.")

        run_name_input = st.text_input(
            "Run name (short description/goal)",
            placeholder="e.g. baseline recursive strategy",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            chunk_sizes_input = st.text_input("Chunk sizes (comma-separated)", "128, 256, 512, 1024")
        with col2:
            overlap = st.number_input("Chunk overlap", min_value=0, value=120)
        with col3:
            strategy = st.selectbox("Chunking strategy", ["recursive", "fixed", "semantic", "sentence"])

        submitted = st.form_submit_button("Run Phase 1 Eval")

    if submitted:
        _execute_phase1(run_name_input, chunk_sizes_input, overlap, strategy, results_dir, config_path)


def _execute_phase1(run_name_input, chunk_sizes_input, overlap, strategy, results_dir, config_path):
    """Execute Phase 1 eval run."""
    try:
        chunk_sizes = [int(x.strip()) for x in chunk_sizes_input.split(",")]
    except ValueError:
        st.error("Invalid chunk sizes. Enter comma-separated integers.")
        st.stop()

    import yaml
    from src.ingest import load_docs
    from src.eval_retrieval import _build_temp_index, _run_retrieval_for_config, _build_reference_content_map

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open("eval/ground_truth.json", "r") as f:
        ground_truth = json.load(f)

    from dotenv import load_dotenv
    load_dotenv()
    from src.config_manager import get_config
    cfg_obj = get_config(config_path)
    embeddings = cfg_obj.get_embeddings()

    progress = st.progress(0, text="Loading documents...")
    docs = load_docs()
    if not docs:
        st.error("No documents found in data/raw/")
        st.stop()

    ref_content_map = _build_reference_content_map(docs, config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name_input.strip() if run_name_input.strip() else \
        f"{strategy}_{'_'.join(str(s) for s in chunk_sizes)}_overlap{overlap}"
    display_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    display_name = f"{run_name} ({display_ts})"

    results = []
    for i, cs in enumerate(chunk_sizes):
        progress.progress(i / len(chunk_sizes), text=f"Evaluating chunk_size={cs}...")
        vs, chunks = _build_temp_index(docs, cs, overlap, strategy, embeddings)
        metrics = _run_retrieval_for_config(vs, ground_truth, config.get("top_k", 4), ref_content_map)
        metrics.update({
            "chunk_size": cs,
            "chunk_overlap": overlap,
            "chunking_strategy": strategy,
            "num_chunks": len(chunks),
            "top_k": config.get("top_k", 4),
            "timestamp": timestamp,
            "run_name": run_name,
            "display_name": display_name,
        })
        results.append(metrics)

        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, f"retrieval_{cs}_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2)

    progress.progress(1.0, text="Done!")
    st.success(f"Saved {len(results)} results — **{display_name}**")
    st.info("Refresh the page to see new results in the charts above.")


def _render_phase2_form(ret_df_raw, all_strategies, results_dir, config_path):
    """Render Phase 2 generation eval form."""
    with st.form("run_phase2_form"):
        st.subheader("Run Phase 2: Generation Eval")
        st.markdown("Run LLM generation on top configs from Phase 1 and evaluate answer quality.")

        p2_run_name = st.text_input(
            "Run name",
            placeholder="e.g. testing semantic generation",
            key="p2_run_name",
        )

        p2_col1, p2_col2 = st.columns(2)
        with p2_col1:
            p2_strategy_filter = st.selectbox(
                "Filter Phase 1 configs by strategy",
                ["all"] + all_strategies,
                key="p2_strategy",
            )
        with p2_col2:
            p2_top_n = st.number_input("Top N configs to evaluate", min_value=1, value=4, key="p2_top_n")

        # Show which Phase 1 configs will be used
        if not ret_df_raw.empty:
            preview_df = ret_df_raw.copy()
            if p2_strategy_filter != "all":
                preview_df = preview_df[preview_df["chunking_strategy"] == p2_strategy_filter]
            if not preview_df.empty:
                preview_df = preview_df.sort_values("timestamp", ascending=False).drop_duplicates(
                    subset=["chunk_size", "chunking_strategy"], keep="first"
                )
                preview_df = preview_df.sort_values("hit_rate", ascending=False).head(p2_top_n)
                st.markdown("**Phase 1 configs that will be used:**")
                st.dataframe(
                    preview_df[["chunking_strategy", "chunk_size", "chunk_overlap",
                                "hit_rate", "mrr", "display_name"]].reset_index(drop=True),
                    use_container_width=True,
                )
            else:
                st.warning("No Phase 1 results found for the selected strategy.")

        p2_submitted = st.form_submit_button("Run Phase 2 Eval")

    if p2_submitted:
        _execute_phase2(p2_run_name, p2_strategy_filter, p2_top_n, results_dir, config_path)


def _execute_phase2(p2_run_name, strategy_filter, top_n, results_dir, config_path):
    """Execute Phase 2 eval run."""
    import yaml
    from src.ingest import load_docs, assign_chunk_ids
    from src.splitters import make_text_splitter
    from src.eval_generation import _load_phase1_results, _cache_key, _load_cache, _save_cache
    from judge.adapters import LangChainJudgeClient
    from judge.scorers import correctness_score, faithfulness_score, relevance_score

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("eval", {})
    cache_dir = eval_cfg.get("cache_dir", "eval/cache")
    cache_enabled = eval_cfg.get("cache_responses", True)

    with open("eval/ground_truth.json", "r") as f:
        ground_truth = json.load(f)

    from dotenv import load_dotenv
    load_dotenv()
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from src.config_manager import get_config
    cfg_obj = get_config(config_path)
    embeddings = cfg_obj.get_embeddings()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge_client = LangChainJudgeClient(llm)

    phase1_results = _load_phase1_results(results_dir)
    if strategy_filter != "all":
        phase1_results = [r for r in phase1_results if r.get("chunking_strategy") == strategy_filter]

    if not phase1_results:
        st.error("No Phase 1 results found for the selected strategy.")
        st.stop()

    top_configs = phase1_results[:top_n]

    progress = st.progress(0, text="Loading documents...")
    docs = load_docs()
    if not docs:
        st.error("No documents found in data/raw/")
        st.stop()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = p2_run_name.strip() if p2_run_name.strip() else \
        f"gen_{'_'.join(str(c['chunk_size']) for c in top_configs)}"
    display_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    display_name = f"{run_name} ({display_ts})"

    total_steps = len(top_configs) * len(ground_truth)
    step = 0

    for cfg_result in top_configs:
        chunk_size = cfg_result["chunk_size"]
        chunk_overlap = cfg_result.get("chunk_overlap", 120)
        chunking_strategy = cfg_result.get("chunking_strategy", "recursive")
        top_k = cfg_result.get("top_k", 4)

        splitter_cfg = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunking_strategy": chunking_strategy,
        }
        splitter = make_text_splitter(splitter_cfg)
        chunks = splitter.split_documents(docs)
        assign_chunk_ids(chunks)
        vs = FAISS.from_documents(chunks, embeddings)
        retriever = vs.as_retriever(search_kwargs={"k": top_k})

        per_question = []
        cache_hits = 0

        for item in ground_truth:
            step += 1
            progress.progress(step / total_steps, text=f"chunk_size={chunk_size}: {item['question'][:40]}...")
            question = item["question"]
            expected_answer = item.get("expected_answer", "")

            ret_docs = retriever.invoke(question)
            retrieved_ids = [d.metadata.get("chunk_id", "") for d in ret_docs]
            context_chunks = [d.page_content for d in ret_docs]
            key = _cache_key(question, retrieved_ids)

            cached = _load_cache(cache_dir, key) if cache_enabled else None

            if cached:
                answer = cached["answer"]
                cache_hits += 1
            else:
                context_text = "\n\n".join(f"[{j+1}] {chunk}" for j, chunk in enumerate(context_chunks))
                prompt = f"""Answer the question using ONLY the provided context. If the context
does not contain the answer, say "I don't have enough information."

Context:
{context_text}

Question: {question}

Answer:"""
                response = llm.invoke(prompt)
                answer = response.content.strip()
                if cache_enabled:
                    _save_cache(cache_dir, key, {
                        "question": question,
                        "answer": answer,
                        "retrieved_chunk_ids": retrieved_ids,
                    })

            correctness = correctness_score(answer, expected_answer, judge_client)
            faithfulness = faithfulness_score(answer, context_chunks, judge_client)
            relevance = relevance_score(question, answer, judge_client)

            per_question.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": answer,
                "retrieved_chunk_ids": retrieved_ids,
                "correctness": correctness,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "cache_hit": cached is not None,
            })

        n = len(per_question)
        config_result = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunking_strategy": chunking_strategy,
            "num_chunks": len(chunks),
            "top_k": top_k,
            "timestamp": timestamp,
            "run_name": run_name,
            "display_name": display_name,
            "cache_hits": cache_hits,
            "avg_correctness": sum(q["correctness"] for q in per_question) / n if n else 0.0,
            "avg_faithfulness": sum(q["faithfulness"] for q in per_question) / n if n else 0.0,
            "avg_relevance": sum(q["relevance"] for q in per_question) / n if n else 0.0,
            "per_question": per_question,
            "phase1_hit_rate": cfg_result.get("hit_rate", 0.0),
            "phase1_mrr": cfg_result.get("mrr", 0.0),
        }

        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, f"generation_{chunk_size}_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(config_result, f, indent=2)

    progress.progress(1.0, text="Done!")
    st.success(f"Phase 2 complete — **{display_name}**")
    st.info("Refresh the page to see new results in the charts above.")
