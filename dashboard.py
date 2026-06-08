import os
import pandas as pd
import streamlit as st


OUTPUT_DIR = "outputs"

METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.csv")
RETRIEVAL_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "retrieval_summary.csv")
RETRIEVAL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "retrieval_results.csv")

CHUNK_FILES = {
    "fixed": os.path.join(OUTPUT_DIR, "fixed_chunks.csv"),
    "overlap": os.path.join(OUTPUT_DIR, "overlap_chunks.csv"),
    "structure": os.path.join(OUTPUT_DIR, "structure_chunks.csv"),
}


st.set_page_config(
    page_title="Data Processing for ChatBots",
    layout="wide"
)


def load_csv(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_csv(path)


def load_all_chunks():
    frames = []

    for strategy, path in CHUNK_FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["strategy"] = strategy
            frames.append(df)

    if not frames:
        st.error("No chunk files found. Run: python src/run_pipeline.py")
        st.stop()

    return pd.concat(frames, ignore_index=True)


def get_strategy_value(df, strategy, column):
    return df.loc[df["strategy"] == strategy, column].iloc[0]


metrics = load_csv(METRICS_FILE)
retrieval_summary = load_csv(RETRIEVAL_SUMMARY_FILE)
retrieval_results = load_csv(RETRIEVAL_RESULTS_FILE)
chunks = load_all_chunks()

combined_results = metrics.merge(
    retrieval_summary,
    on="strategy",
    how="left"
)

best_accuracy = combined_results["retrieval_accuracy_percent"].max()

best_accuracy_rows = combined_results[
    combined_results["retrieval_accuracy_percent"] == best_accuracy
].copy()

best_tradeoff_row = best_accuracy_rows.sort_values(
    by=["storage_size_kb", "number_of_chunks", "processing_time_seconds"],
    ascending=[True, True, True]
).iloc[0]

best_tradeoff_strategy = best_tradeoff_row["strategy"]
best_tradeoff_storage = best_tradeoff_row["storage_size_kb"]
best_tradeoff_chunks = int(best_tradeoff_row["number_of_chunks"])

lowest_storage = metrics["storage_size_kb"].min()
lowest_storage_strategy = metrics.loc[
    metrics["storage_size_kb"].idxmin(),
    "strategy"
]

lowest_chunks = int(metrics["number_of_chunks"].min())
lowest_chunks_strategy = metrics.loc[
    metrics["number_of_chunks"].idxmin(),
    "strategy"
]

metadata_completeness = round(
    metrics["metadata_completeness_percent"].mean(), 2
)


st.title("Data Processing for ChatBots")
st.subheader("Dashboard: chunking strategies, retrieval quality and governance")

st.markdown(
    """
    This dashboard presents the results of an experiment comparing three data processing
    strategies for chatbot knowledge bases: **fixed chunking**, **overlapping chunking**
    and **structure-aware chunking**.
    """
)


st.header("1. Experiment results")
st.dataframe(metrics, use_container_width=True)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Best retrieval accuracy", f"{best_accuracy:.1f}%")
col2.metric("Best trade-off strategy", best_tradeoff_strategy)
col3.metric("Lowest storage", f"{lowest_storage:.2f} KB", lowest_storage_strategy)
col4.metric("Fewest chunks", lowest_chunks, lowest_chunks_strategy)
col5.metric("Metadata completeness", f"{metadata_completeness:.1f}%")


st.info(
    f"Best trade-off was selected dynamically: first the dashboard finds the strategy "
    f"with the highest retrieval accuracy, and if more than one strategy has the same "
    f"accuracy, it selects the one with the lowest storage size. In this run, "
    f"**{best_tradeoff_strategy}** was selected with **{best_accuracy:.1f}% accuracy**, "
    f"**{best_tradeoff_storage:.2f} KB storage**, and **{best_tradeoff_chunks} chunks**."
)


st.header("2. Number of chunks by strategy")
st.bar_chart(metrics.set_index("strategy")["number_of_chunks"])

max_chunks = int(metrics["number_of_chunks"].max())
max_chunks_strategy = metrics.loc[
    metrics["number_of_chunks"].idxmax(),
    "strategy"
]

st.markdown(
    f"""
    The strategy with the highest number of chunks was **{max_chunks_strategy}**
    with **{max_chunks} chunks**. More chunks usually mean a larger index,
    more embedding operations and higher storage cost.
    """
)


st.header("3. Average chunk length")
st.bar_chart(metrics.set_index("strategy")["average_chunk_length"])

shortest_avg_length = metrics["average_chunk_length"].min()
shortest_avg_strategy = metrics.loc[
    metrics["average_chunk_length"].idxmin(),
    "strategy"
]

st.markdown(
    f"""
    The shortest average chunk length was produced by **{shortest_avg_strategy}**
    with an average of **{shortest_avg_length:.2f} characters**.
    Shorter chunks can improve interpretability and lineage, because each chunk
    is easier to map back to a specific part of the original document.
    """
)


st.header("4. Storage size by strategy")
st.bar_chart(metrics.set_index("strategy")["storage_size_kb"])

highest_storage = metrics["storage_size_kb"].max()
highest_storage_strategy = metrics.loc[
    metrics["storage_size_kb"].idxmax(),
    "strategy"
]

st.markdown(
    f"""
    The lowest storage usage was achieved by **{lowest_storage_strategy}**
    with **{lowest_storage:.2f} KB**. The highest storage usage was produced by
    **{highest_storage_strategy}** with **{highest_storage:.2f} KB**.
    This metric represents the approximate size of the generated chunk index files.
    """
)


st.header("5. Processing time by strategy")
st.bar_chart(metrics.set_index("strategy")["processing_time_seconds"])

fastest_time = metrics["processing_time_seconds"].min()
fastest_strategy = metrics.loc[
    metrics["processing_time_seconds"].idxmin(),
    "strategy"
]

st.markdown(
    f"""
    The fastest strategy in this run was **{fastest_strategy}**
    with **{fastest_time:.4f} seconds**. Processing times are very low because
    the dataset is small, but the metric becomes more important at larger scale.
    """
)


st.header("6. Retrieval accuracy")
st.dataframe(retrieval_summary, use_container_width=True)
st.bar_chart(retrieval_summary.set_index("strategy")["retrieval_accuracy_percent"])

accuracy_lines = []

for _, row in retrieval_summary.iterrows():
    accuracy_lines.append(
        f"- **{row['strategy']}** achieved **{row['retrieval_accuracy_percent']:.1f}%** "
        f"retrieval accuracy ({int(row['correct_answers'])}/{int(row['total_questions'])} correct answers)."
    )

st.markdown(
    "\n".join(accuracy_lines)
)

best_strategy_list = ", ".join(best_accuracy_rows["strategy"].tolist())

st.markdown(
    f"""
    The highest retrieval accuracy was **{best_accuracy:.1f}%**, achieved by:
    **{best_strategy_list}**.

    These results show how chunking strategy can influence retrieval quality.
    """
)


st.header("7. Retrieval evidence")

selected_strategy = st.selectbox(
    "Select strategy",
    retrieval_results["strategy"].unique()
)

filtered_results = retrieval_results[
    retrieval_results["strategy"] == selected_strategy
]

st.dataframe(
    filtered_results[
        [
            "question",
            "expected_document",
            "retrieved_document",
            "correct",
            "similarity_score",
            "owner",
            "classification",
        ]
    ],
    use_container_width=True
)

selected_question = st.selectbox(
    "Select question",
    filtered_results["question"].unique()
)

selected_row = filtered_results[
    filtered_results["question"] == selected_question
].iloc[0]

st.markdown("### Retrieved chunk")
st.write(selected_row["retrieved_chunk_text"])

st.markdown("### Governance metadata")
st.json({
    "chunk_id": selected_row["chunk_id"],
    "source_file": selected_row["source_file"],
    "owner": selected_row["owner"],
    "classification": selected_row["classification"],
})


st.header("8. Data lineage")

lineage_columns = [
    "strategy",
    "chunk_id",
    "document_id",
    "source_file",
    "owner",
    "classification",
    "version",
    "processed_at",
]

existing_columns = [col for col in lineage_columns if col in chunks.columns]
lineage_table = chunks[existing_columns].copy()

st.dataframe(lineage_table, use_container_width=True)

st.markdown(
    """
    The lineage table shows how every chunk can be traced back to its source document,
    owner, version and classification. This is the governance layer of the project.
    """
)


st.header("9. Final comparison and recommendation")

comparison = combined_results[[
    "strategy",
    "number_of_chunks",
    "storage_size_kb",
    "processing_time_seconds",
    "retrieval_accuracy_percent",
    "metadata_completeness_percent",
]].copy()

comparison["recommendation"] = comparison["strategy"].apply(
    lambda strategy: (
        "Best trade-off in this experiment"
        if strategy == best_tradeoff_strategy
        else "Useful depending on cost, quality and governance requirements"
    )
)

st.dataframe(comparison, use_container_width=True)

st.markdown(
    f"""
    **Main conclusion:** data processing strategy has a direct impact on chatbot
    knowledge base quality, storage size and retrieval performance.

    In this experiment, the best trade-off strategy was **{best_tradeoff_strategy}**.
    It was selected because it achieved the highest retrieval accuracy and, among
    the strategies with the highest accuracy, required the lowest storage.

    This means that the best strategy is not simply the one with the smallest index
    or the one with the most detailed chunking. The best choice depends on a balance
    between retrieval quality, storage cost, processing time and governance needs.
    """
)