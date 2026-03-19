# LLM Data Quality Guardian

A production-grade data quality monitoring pipeline purpose-built for LLM and RAG systems. It goes beyond traditional null checks and schema validation to detect **semantic drift**, **embedding distribution shifts**, and **hallucination-triggering patterns** in your knowledge base.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![100% Free Stack](https://img.shields.io/badge/cost-%240-brightgreen)

---

## The Problem

Traditional data quality tools (Great Expectations, Soda Core, dbt tests) validate schemas, nulls, and row counts. But LLM and RAG systems fail in ways those tools cannot catch:

- Two documents say opposite things about the same topic, and the LLM picks the wrong one
- Your knowledge base silently drifts in meaning over weeks as new content is added
- Outdated facts sit alongside current ones with no expiration mechanism
- Entire topic areas have too few documents for reliable retrieval
- Retrieved chunks score high on similarity but actually answer a different question

This project monitors all of those failure modes automatically, using statistical tests on embedding distributions and NLP-based pattern detection.

## What This Demonstrates

| Skill | How It Shows Up |
|-------|----------------|
| **Data Engineering** | Multi-format ingestion (CSV, JSON, text), pipeline orchestration, metrics persistence |
| **ML/NLP Engineering** | Embedding generation, cosine similarity, KS tests, MMD kernel methods |
| **LLM/RAG Expertise** | Retrieval quality scoring, chunk analysis, context coverage, hallucination pattern detection |
| **Software Design** | Strategy pattern, Pydantic v2 data contracts, modular architecture, CLI + dashboard |
| **Testing** | Full pytest suite covering drift, hallucination, ingestion, quality, RAG, and storage |
| **Observability** | Time-series metrics, alerting (console + webhook), Streamlit dashboard with Plotly |

---

## Architecture

```
+----------------------------------------------------------+
|                    Guardian Pipeline                      |
|                                                          |
|  +----------+  +------------+  +----------------------+  |
|  | Ingestion|  | Embeddings |  |   Detection Layer    |  |
|  | CSV/JSON |->| MiniLM-L6  |->| +------+ +--------+ |  |
|  | Text/MD  |  | + Caching  |  | |Drift | |Halluc. | |  |
|  +----------+  +------------+  | |KS/MMD| |Contra. | |  |
|                                | |Cosine| |Temporal| |  |
|  +----------+  +------------+  | +------+ |Sparse  | |  |
|  | ChromaDB |  |  SQLite    |  |          |Ambig.  | |  |
|  | Vectors  |  |  Metrics   |  | +------+ +--------+ |  |
|  +----------+  +------------+  | |Soda  | +--------+ |  |
|                                | |Core  | |  RAG   | |  |
|  +----------+  +------------+  | |+Custom| |Quality | |  |
|  |  Alerts  |  | Streamlit  |  | +------+ +--------+ |  |
|  |Console/WH|  | Dashboard  |  +----------------------+  |
|  +----------+  +------------+                            |
+----------------------------------------------------------+
```

## Tech Stack (100% Free)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embeddings | sentence-transformers (MiniLM-L6-v2) | Local embedding generation, no API keys needed |
| Vector Store | ChromaDB | Document storage and similarity search |
| Quality Checks | Soda Core + custom checks | Traditional + semantic data quality validation |
| Drift Detection | SciPy + scikit-learn + NumPy | KS test, MMD, cosine centroid shift |
| Metrics DB | SQLite | Time-series metrics persistence |
| Dashboard | Streamlit + Plotly | Interactive visualization with 5 monitoring tabs |
| Data Contracts | Pydantic v2 | Type-safe models across the entire pipeline |
| CLI | argparse + Rich | Terminal interface with formatted output |

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/ShishirN21/LLM-Data-Quality-Guardian.git
cd LLM-Data-Quality-Guardian
pip install -e ".[dev]"
```

### 2. Run the Pipeline

```bash
guardian run --config config/default.yaml
```

This will:
1. Ingest sample documents (CSV, JSON, and plain text)
2. Generate embeddings using MiniLM-L6-v2
3. Store vectors in ChromaDB
4. Run drift detection (KS test, MMD, cosine centroid)
5. Scan for hallucination-triggering patterns
6. Execute data quality checks
7. Evaluate RAG retrieval quality
8. Output a summary table and fire alerts

### 3. View Results

```bash
# Terminal report with color-coded results
guardian report

# Launch the interactive Streamlit dashboard
guardian dashboard
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Use Cases

### Use Case 1: Catching Contradictions Before They Cause Hallucinations

The sample dataset intentionally contains a contradiction:

- **doc_018**: "Neural networks are not inspired by biological neurons."
- **doc_019**: "Neural networks are inspired by biological neurons in the human brain."

The contradiction detector finds document pairs with high semantic similarity but opposing claims. In a production RAG system, if both documents are retrieved as context, the LLM may hallucinate or give inconsistent answers depending on which chunk it prioritizes.

### Use Case 2: Detecting Stale Knowledge

**doc_025** discusses Python 2 end-of-life and was written in June 2019. The temporal staleness detector flags documents older than a configurable threshold (default: 365 days), because outdated facts are one of the most common sources of LLM hallucination in enterprise RAG systems.

### Use Case 3: Monitoring Embedding Distribution Drift

When new batches of documents are ingested over time, the embedding distribution can shift. For example, if a knowledge base about "databases" gradually starts including more content about "DevOps," the embedding centroid moves. The pipeline runs three complementary drift tests:

- **Kolmogorov-Smirnov (KS) test**: Per-dimension distribution comparison with configurable alpha
- **Maximum Mean Discrepancy (MMD)**: Full distribution-level comparison using an RBF kernel
- **Cosine centroid shift**: Measures how much the mean embedding vector has moved

If any test exceeds its threshold, an alert fires.

### Use Case 4: Identifying Sparse Knowledge Regions

The sparse context detector identifies topic areas where the knowledge base has too few documents clustered together. When a user asks a question in a sparse region, the RAG system retrieves low-relevance chunks and the LLM is forced to "fill in the gaps," leading to hallucinations.

### Use Case 5: RAG Retrieval Quality Monitoring

The pipeline runs sample queries against ChromaDB and measures:

- **Retrieval relevance**: Are the top-K results actually semantically relevant to the query?
- **Chunk quality**: Are document chunks sized appropriately for the context window?
- **Context coverage**: Do retrieved documents collectively cover the query topic?
- **Redundancy**: Are retrieved chunks too similar to each other, wasting context window space?

---

## How Detection Works

### Semantic Drift Detection

Three statistical tests run on every pipeline execution, comparing the current batch of embeddings against the historical baseline stored in ChromaDB:

| Test | What It Measures | When It Fires |
|------|-----------------|---------------|
| KS Test | Per-dimension distribution shift | p-value < alpha (default 0.05) |
| MMD | Distribution-level divergence via RBF kernel | MMD^2 > threshold (default 0.1) |
| Cosine Centroid | Mean vector direction change | Similarity < threshold (default 0.85) |

### Hallucination Risk Patterns

Four detectors scan the document corpus for patterns known to trigger LLM hallucinations:

| Detector | What It Finds | Example |
|----------|--------------|---------|
| Contradiction | Semantically similar docs with conflicting facts | "X is true" vs "X is not true" |
| Entity Ambiguity | Same entity name used with different meanings | "Python" (language) vs "Python" (snake) |
| Temporal Staleness | Documents with outdated information | Facts from 2019 still in the knowledge base |
| Sparse Context | Topic areas with insufficient document coverage | A single doc on a niche topic with no supporting docs |

### Traditional Data Quality

- Row count validation
- Null and missing value detection
- Duplicate content detection
- Document length distribution analysis
- Content emptiness checks

---

## Configuration

All settings are in `config/default.yaml`. Key parameters:

```yaml
drift:
  ks_test_alpha: 0.05          # Significance level for KS test
  cosine_sim_threshold: 0.85   # Below this = drift detected
  mmd_threshold: 0.1           # MMD^2 above this = drift detected
  min_samples: 10              # Minimum docs before drift runs

hallucination:
  contradiction_sim_threshold: 0.8   # Similarity to flag contradictions
  temporal_staleness_days: 365       # Days before content is "stale"
  sparse_context_min_neighbors: 5    # Minimum neighbors for coverage

rag:
  relevance_threshold: 0.7    # Minimum query-doc similarity
  chunk_size_target: 512      # Target chunk size in tokens
  top_k: 5                    # Number of results to retrieve
```

## Dashboard

The Streamlit dashboard provides five tabs for monitoring:

1. **Overview** -- Health score gauge, key metrics, recent alerts
2. **Drift Detection** -- Time-series of drift statistics, pass/fail indicators
3. **Hallucination Risks** -- Risk type distribution, severity breakdown, detailed table
4. **Data Quality** -- Pass/fail badges, historical pass rate trends
5. **RAG Quality** -- Retrieval relevance scores, chunk analysis, coverage metrics

Launch it with:

```bash
guardian dashboard
```

---

## Project Structure

```
guardian/
├── models.py              # Pydantic v2 data contracts (Document, DriftResult, Alert, etc.)
├── pipeline.py            # Main orchestrator tying all modules together
├── cli.py                 # CLI with run, report, and dashboard commands
├── ingestion/             # Multi-format loaders (CSV, JSON, text)
├── embeddings/            # Sentence-transformer wrapper with disk caching
├── drift/                 # KS test, MMD, cosine centroid (Strategy pattern)
├── hallucination/         # Contradiction, ambiguity, temporal, sparse detectors
├── quality/               # Soda Core integration + custom quality checks
├── rag/                   # Retrieval scoring, chunking, context coverage
├── storage/               # SQLite metrics store + ChromaDB vector store
├── alerts/                # Console + webhook alert handlers
└── dashboard/             # Streamlit app with Plotly visualizations

tests/
├── test_drift.py          # Drift detection unit tests
├── test_hallucination.py  # Hallucination pattern tests
├── test_ingestion.py      # Loader tests for all formats
├── test_quality.py        # Quality check tests
├── test_rag.py            # RAG pipeline tests
└── test_storage.py        # Metrics store tests

config/
└── default.yaml           # All configurable thresholds and parameters

data/sample/               # Sample dataset with intentional quality issues
├── documents.csv          # 25 docs with contradictions and stale facts
├── facts.json             # Structured knowledge base entries
├── notes.txt              # Unstructured text documents
└── queries.txt            # Sample queries for RAG evaluation
```

---

## Extending the Pipeline

**Add a new data source**: Implement `BaseLoader` in `guardian/ingestion/` and register it in the loader registry.

**Add a new drift test**: Implement the `DriftStrategy` abstract class in `guardian/drift/` and add it to `DriftDetector.__init__`.

**Add a new hallucination detector**: Implement the `PatternDetector` abstract class in `guardian/hallucination/` and register it in `HallucinationDetector.__init__`.

**Add webhook alerts**: Set `webhook_enabled: true` and provide a `webhook_url` in `config/default.yaml` to send alerts to Slack, PagerDuty, or any HTTP endpoint.

---

## Why This Project Exists

In 2025, the industry shifted heavily toward monitoring LLM accuracy, including RAG pipeline outputs. Tools like Monte Carlo, WhyLabs, and Galileo lead this space commercially, but most data engineering portfolios do not touch semantic-level data quality. This project fills that gap with a fully open-source, zero-cost implementation that demonstrates practical understanding of:

- How embedding distributions behave over time
- Why contradictions and stale data cause hallucinations
- How to measure RAG retrieval quality programmatically
- How to build observable, testable ML pipelines

---

## License

MIT
