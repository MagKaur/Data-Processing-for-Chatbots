import os
import re
import time
from datetime import datetime

import pandas as pd


RAW_DIR = "data/raw"
OUTPUT_DIR = "outputs"

FIXED_SIZE = 500
OVERLAP_SIZE = 100


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_metadata(text, filename):
    metadata = {
        "document_id": filename.replace(".txt", ""),
        "source_file": filename,
        "title": "",
        "owner": "Unknown",
        "version": "Unknown",
        "classification": "Unknown",
    }

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("Title:"):
            metadata["title"] = line.replace("Title:", "").strip()

        elif line.startswith("Owner:"):
            metadata["owner"] = line.replace("Owner:", "").strip()

        elif line.startswith("Version:"):
            metadata["version"] = line.replace("Version:", "").strip()

        elif line.startswith("Classification:"):
            metadata["classification"] = line.replace("Classification:", "").strip()

    return metadata


def load_documents():
    documents = []

    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(RAW_DIR, filename)

            with open(path, "r", encoding="utf-8") as file:
                raw_text = file.read()

            metadata = extract_metadata(raw_text, filename)

            documents.append({
                **metadata,
                "raw_text": raw_text,
                "clean_text": clean_text(raw_text),
            })

    return documents


def fixed_chunking(text, size=FIXED_SIZE):
    chunks = []

    for start in range(0, len(text), size):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def overlapping_chunking(text, size=FIXED_SIZE, overlap=OVERLAP_SIZE):
    chunks = []
    step = size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def structure_aware_chunking(raw_text):
    """
    Dzieli dokument po sekcjach zaczynających się od 'Section:'.
    Dzięki temu chunk odpowiada logicznej części dokumentu.
    """
    chunks = []

    parts = re.split(r"(?=Section:)", raw_text)

    header = parts[0].strip()

    for part in parts[1:]:
        chunk = part.strip()

        if chunk:
            chunks.append(chunk)

    if not chunks and header:
        chunks.append(header)

    return chunks


def create_chunks_for_strategy(documents, strategy_name, chunking_function):
    start_time = time.perf_counter()

    records = []

    for doc in documents:
        if strategy_name == "structure":
            chunks = chunking_function(doc["raw_text"])
        else:
            chunks = chunking_function(doc["clean_text"])

        for index, chunk_text in enumerate(chunks, start=1):
            records.append({
                "strategy": strategy_name,
                "chunk_id": f"{strategy_name}_{doc['document_id']}_chunk_{index}",
                "document_id": doc["document_id"],
                "source_file": doc["source_file"],
                "title": doc["title"],
                "owner": doc["owner"],
                "version": doc["version"],
                "classification": doc["classification"],
                "chunk_number": index,
                "chunk_text": chunk_text,
                "chunk_length": len(chunk_text),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    processing_time = time.perf_counter() - start_time

    chunks_df = pd.DataFrame(records)

    return chunks_df, processing_time


def calculate_metrics(strategy_name, chunks_df, processing_time):
    output_file = os.path.join(OUTPUT_DIR, f"{strategy_name}_chunks.csv")

    chunks_df.to_csv(output_file, index=False)

    storage_size_kb = os.path.getsize(output_file) / 1024

    required_metadata_columns = [
        "chunk_id",
        "document_id",
        "source_file",
        "owner",
        "version",
        "classification",
        "processed_at",
    ]

    metadata_completeness = (
        chunks_df[required_metadata_columns]
        .notnull()
        .mean()
        .mean()
        * 100
    )

    duplicate_chunks = chunks_df["chunk_text"].duplicated().sum()

    metrics = {
        "strategy": strategy_name,
        "number_of_chunks": len(chunks_df),
        "average_chunk_length": round(chunks_df["chunk_length"].mean(), 2),
        "duplicate_chunks": int(duplicate_chunks),
        "storage_size_kb": round(storage_size_kb, 2),
        "processing_time_seconds": round(processing_time, 4),
        "metadata_completeness_percent": round(metadata_completeness, 2),
    }

    return metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")

    strategies = {
        "fixed": fixed_chunking,
        "overlap": overlapping_chunking,
        "structure": structure_aware_chunking,
    }

    all_metrics = []

    for strategy_name, chunking_function in strategies.items():
        print(f"\nRunning strategy: {strategy_name}")

        chunks_df, processing_time = create_chunks_for_strategy(
            documents,
            strategy_name,
            chunking_function
        )

        metrics = calculate_metrics(
            strategy_name,
            chunks_df,
            processing_time
        )

        all_metrics.append(metrics)

        print(f"Chunks created: {metrics['number_of_chunks']}")
        print(f"Average chunk length: {metrics['average_chunk_length']}")
        print(f"Storage size: {metrics['storage_size_kb']} KB")
        print(f"Processing time: {metrics['processing_time_seconds']} s")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

    print("\nPipeline finished.")
    print("Generated files:")
    print("- outputs/fixed_chunks.csv")
    print("- outputs/overlap_chunks.csv")
    print("- outputs/structure_chunks.csv")
    print("- outputs/metrics.csv")


if __name__ == "__main__":
    main()