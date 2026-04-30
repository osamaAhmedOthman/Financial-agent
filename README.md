# AI Financial Auditor

AI Financial Auditor is an end-to-end financial document auditing platform built for Egyptian tax workflows. It combines document ingestion, deterministic validation, knowledge-base retrieval, and LLM-assisted reasoning to produce structured compliance reports with citations, risk scoring, and review recommendations.

## What The Project Does

The application is designed to:

- Ingest invoices, statements, and tax-related documents from PDF, images, and Excel files.
- Extract and normalize structured financial data.
- Validate critical arithmetic and tax rules with deterministic checks.
- Retrieve relevant legal context from an Egyptian tax law knowledge base.
- Generate an audit report with violations, recommendations, confidence, and human-review flags.
- Present the result in a Streamlit UI with history and export support.

## End-to-End Flow

1. The user uploads a document in the Streamlit app.
2. The ingestion pipeline detects the document type and extracts the relevant fields.
3. Validation rules check tax ID format, VAT logic, and financial consistency.
4. The agent builds legal queries from the extracted content.
5. The knowledge base returns matching legal articles using hybrid retrieval.
6. The auditor produces a structured report with compliance scoring and legal references.
7. The UI displays the result and stores the run in session history.

## Project Structure

```text
Financial-agent/
├── agent/                # LangGraph auditing agent and output schemas
├── ingestion/            # Document classification, extraction, and validation
├── knowledge_base/       # Legal chunking, embedding, and retrieval
├── tests/                # Unit and integration tests
├── ui/                   # Streamlit application and pages
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build for the UI
└── README.md             # Project guide
```

## Main Components

### UI

- `ui/app.py` is the main Streamlit entry point.
- `ui/pages/1_upload.py` handles file upload and ingestion.
- `ui/pages/2_audit.py` triggers the audit workflow.
- `ui/pages/3_report.py` renders the final compliance report.
- `ui/pages/4_history.py` shows past runs for the current session.
- `ui/utils.py` contains shared formatting and PDF export helpers.

### Ingestion

- `ingestion/classifiers.py` classifies document type.
- `ingestion/extractors.py` extracts text from PDFs, images, and spreadsheets.
- `ingestion/llm_extractor.py` applies extraction logic with LLM fallback.
- `ingestion/validators.py` applies deterministic validation rules.
- `ingestion/processors.py` orchestrates the ingestion pipeline.

### Agent

- `agent/graph.py` defines the LangGraph workflow.
- `agent/tools.py` provides legal retrieval and web search helpers.
- `agent/auditor.py` exposes the public audit entry points.
- `agent/schemas.py` defines the final audit report models.

### Knowledge Base

- `knowledge_base/chunker.py` splits legal text into article-level chunks.
- `knowledge_base/embedder.py` embeds and upserts chunks into Pinecone.
- `knowledge_base/retriever.py` performs hybrid retrieval using vector search and BM25.
- `knowledge_base/loader.py` loads seed data or PDFs into the knowledge base.

## Requirements

- Python 3.10+
- Pinecone API key for knowledge-base storage and retrieval
- Groq API key for agent reasoning
- Tavily API key if web search is enabled
- Streamlit for the UI

## Environment Setup

Create a `.env` file from the provided example and fill in your keys:

```bash
copy .env.example .env
```

Important variables:

- `GROQ_API_KEY`
- `PINECONE_API_KEY`
- `TAVILY_API_KEY`
- `OPENAI_API_KEY` if you use OpenAI-backed components
- `LANGCHAIN_API_KEY` and `LANGCHAIN_PROJECT` for tracing

The repository ignores `.env`, so your secrets will not be committed.

## Local Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run The App

Start the Streamlit UI with:

```bash
streamlit run ui/app.py
```

## Run The Tests

Run the full test suite with:

```bash
pytest tests -q
```

Current test behavior:

- Unit tests run locally without external services.
- Integration tests are marked and will skip automatically when Pinecone, Groq, or the embedding stack is unavailable.

## Docker

The repository includes a `Dockerfile` for containerized runs of the UI.

Build the image:

```bash
docker build -t ai-financial-auditor .
```

Run the container:

```bash
docker run --rm -p 8501:8501 --env-file .env ai-financial-auditor
```

Then open the app at `http://localhost:8501`.

## Notes On Testing And Reliability

- The test suite includes both fast unit tests and guarded integration tests.
- Knowledge-base tests verify chunking, BM25 retrieval, and Pinecone integration behavior.
- Agent tests verify schema validation, tool behavior, and audit flow contracts.
- External-service tests are designed to skip cleanly instead of failing when credentials or runtime dependencies are missing.

## License And Usage

This project is intended for internal or organizational use as a financial-document auditing assistant. Review and adapt the legal and compliance logic before using it in production.
