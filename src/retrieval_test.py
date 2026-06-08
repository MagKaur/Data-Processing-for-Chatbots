import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OUTPUT_DIR = "outputs"

STRATEGY_FILES = {
    "fixed": "outputs/fixed_chunks.csv",
    "overlap": "outputs/overlap_chunks.csv",
    "structure": "outputs/structure_chunks.csv",
}

TEST_QUESTIONS = [
    {
        "question": "What is the remote work policy?",
        "expected_document": "remote_work_policy",
    },
    {
        "question": "How should personal data be processed?",
        "expected_document": "privacy_policy",
    },
    {
        "question": "Who can access confidential documents?",
        "expected_document": "access_control_policy",
    },
    {
        "question": "What is the return policy?",
        "expected_document": "returns_policy",
    },
    {
        "question": "What should employees do after a security incident?",
        "expected_document": "incident_response",
    },
    {
        "question": "How long are business records retained?",
        "expected_document": "data_retention_policy",
    },
    {
        "question": "What are password requirements?",
        "expected_document": "it_security_policy",
    },
    {
        "question": "How can customers track orders?",
        "expected_document": "customer_faq",
    },
]


def load_chunks(strategy_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing file for strategy '{strategy_name}': {file_path}. "
            "Run python src/run_pipeline.py first."
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "strategy",
        "chunk_id",
        "document_id",
        "source_file",
        "owner",
        "classification",
        "chunk_text",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"File {file_path} is missing columns: {missing_columns}"
        )

    df["chunk_text"] = df["chunk_text"].fillna("")

    return df


def retrieve_top_chunk(chunks_df, question):
    documents = chunks_df["chunk_text"].tolist()

    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(documents)
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    best_row = chunks_df.iloc[best_index].copy()
    best_row["similarity_score"] = round(float(best_score), 4)

    return best_row


def evaluate_strategy(strategy_name, file_path):
    chunks_df = load_chunks(strategy_name, file_path)

    results = []

    for test in TEST_QUESTIONS:
        question = test["question"]
        expected_document = test["expected_document"]

        top_result = retrieve_top_chunk(chunks_df, question)

        retrieved_document = top_result["document_id"]
        correct = int(retrieved_document == expected_document)

        results.append({
            "strategy": strategy_name,
            "question": question,
            "expected_document": expected_document,
            "retrieved_document": retrieved_document,
            "correct": correct,
            "similarity_score": top_result["similarity_score"],
            "chunk_id": top_result["chunk_id"],
            "source_file": top_result["source_file"],
            "owner": top_result["owner"],
            "classification": top_result["classification"],
            "retrieved_chunk_text": top_result["chunk_text"],
        })

    return results


def create_accuracy_summary(results_df):
    summary = (
        results_df
        .groupby("strategy")
        .agg(
            total_questions=("question", "count"),
            correct_answers=("correct", "sum"),
            avg_similarity_score=("similarity_score", "mean"),
        )
        .reset_index()
    )

    summary["retrieval_accuracy_percent"] = (
        summary["correct_answers"] / summary["total_questions"] * 100
    ).round(2)

    summary["avg_similarity_score"] = summary["avg_similarity_score"].round(4)

    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    print("Running retrieval evaluation...")

    for strategy_name, file_path in STRATEGY_FILES.items():
        print(f"Testing strategy: {strategy_name}")

        strategy_results = evaluate_strategy(strategy_name, file_path)
        all_results.extend(strategy_results)

    results_df = pd.DataFrame(all_results)
    summary_df = create_accuracy_summary(results_df)

    results_path = os.path.join(OUTPUT_DIR, "retrieval_results.csv")
    summary_path = os.path.join(OUTPUT_DIR, "retrieval_summary.csv")

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nRetrieval evaluation finished.")
    print("\nSummary:")
    print(summary_df)

    print("\nGenerated files:")
    print("- outputs/retrieval_results.csv")
    print("- outputs/retrieval_summary.csv")


if __name__ == "__main__":
    main()