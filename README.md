# Data Processing for ChatBots

## Project Overview

This project investigates how different data processing strategies affect the quality, storage requirements, and governance of chatbot knowledge bases.

The project focuses on the preprocessing stage of chatbot systems, where source documents are transformed into chunks that can later be indexed, embedded, and retrieved by conversational AI systems.

The main objective is to compare different chunking strategies and evaluate their impact on retrieval quality, storage requirements, and governance-related aspects such as metadata management and data lineage.

---

## Research Question

**How do different data processing strategies affect chatbot knowledge base quality, cost, and governance?**

---

## Project Scope

The project implements a complete data processing pipeline:

1. Document ingestion
2. Text cleaning
3. Chunk generation
4. Metadata enrichment
5. Retrieval evaluation
6. Metrics calculation
7. Governance dashboard

Three chunking strategies are compared:

* Fixed Chunking
* Overlapping Chunking
* Structure-Aware Chunking

---

## Dataset

The dataset consists of synthetic enterprise documents representing internal company policies and procedures.

Example documents:

* HR Policy
* Remote Work Policy
* IT Security Policy
* Privacy Policy
* Returns Policy
* Access Control Policy
* Incident Response Procedure
* Data Retention Policy
* VPN Usage Policy
* Data Classification Policy

Each document contains:

* Title
* Owner
* Version
* Classification
* Multiple business sections

---

## Architecture

```text
Raw Documents
      │
      ▼
Document Loading
      │
      ▼
Text Cleaning
      │
      ▼
Chunk Generation
      │
      ├── Fixed Chunking
      ├── Overlapping Chunking
      └── Structure-Aware Chunking
      │
      ▼
Metadata Enrichment
      │
      ▼
Metrics Calculation
      │
      ▼
Retrieval Evaluation
      │
      ▼
Governance Dashboard
```

---

## Implemented Chunking Strategies

### 1. Fixed Chunking

Documents are split into chunks of a fixed size.

Advantages:

* Simple
* Fast
* Low storage requirements

Disadvantages:

* May split sentences and context
* Lower retrieval quality

---

### 2. Overlapping Chunking

Documents are split into chunks with overlapping content.

Advantages:

* Better context preservation
* Improved retrieval quality

Disadvantages:

* Increased storage requirements
* More chunks generated

---

### 3. Structure-Aware Chunking

Documents are split according to document sections.

Advantages:

* Preserves document structure
* Better interpretability
* Improved lineage

Disadvantages:

* Requires structured documents
* More chunks generated

---

## Governance Layer

Each generated chunk contains metadata:

* chunk_id
* document_id
* source_file
* owner
* version
* classification
* processed_at
* strategy

The project demonstrates key Big Data Governance concepts:

### Metadata Management

Every chunk is enriched with governance metadata.

### Data Lineage

Each retrieval result can be traced back to:

Chunk → Document → Source File → Owner

### Data Quality

The project measures:

* Metadata completeness
* Duplicate chunks
* Storage size
* Processing time

---

## Retrieval Evaluation

A retrieval experiment was performed using TF-IDF vectorization and cosine similarity.

Example questions:

* What is the remote work policy?
* How should personal data be processed?
* Who can access confidential documents?
* What is the return policy?
* What should employees do after a security incident?

Retrieval quality is measured as:

```text
Retrieval Accuracy =
Correct Answers / Total Questions
```

---

## Experimental Results

| Strategy  | Chunks | Storage (KB) | Retrieval Accuracy |
| --------- | ------ | ------------ | ------------------ |
| Fixed     | 26     | 12.93        | 75%                |
| Overlap   | 28     | 14.69        | 100%               |
| Structure | 52     | 16.33        | 100%               |

---

## Key Findings

1. Data processing strategy directly impacts retrieval quality.
2. Fixed chunking produces the smallest index and lowest storage cost.
3. Overlapping chunking achieves perfect retrieval accuracy with relatively low storage overhead.
4. Structure-aware chunking preserves document semantics and governance traceability.
5. Governance begins before the language model stage, during data preparation.

---

## Project Structure

```text
DataProcessingForChatBots/
│
├── data/
│   └── raw/
│
├── src/
│   ├── run_pipeline.py
│   └── retrieval_test.py
│
├── outputs/
│   ├── fixed_chunks.csv
│   ├── overlap_chunks.csv
│   ├── structure_chunks.csv
│   ├── metrics.csv
│   ├── retrieval_results.csv
│   └── retrieval_summary.csv
│
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install pandas
pip install scikit-learn
pip install streamlit
pip install matplotlib
```

---

## Running the Project

Generate chunks and metrics:

```bash
python src/run_pipeline.py
```

Run retrieval evaluation:

```bash
python src/retrieval_test.py
```

Start dashboard:

```bash
streamlit run dashboard.py
```

---

## Future Work

Potential future extensions include:

* Vector databases
* Embedding models
* Incremental document updates
* Access control policies
* PII detection
* Data drift monitoring
* Real-world enterprise datasets

---

## Author

Magdalena Kochanowska

Project developed as part of the Big Data Governance course.

Topic: **Data Processing for ChatBots**
