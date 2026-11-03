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
6. The auditor produces a structured report with compliance scoring and legal references, and a deterministic check flags any stated VAT rate that is not a legal rate.
7. The UI displays the result and stores the run in session history.

## Project Structure

```text
Financial-agent/
├── agent/                # LangGraph auditing agent and output schemas
├── ingestion/            # Document classification, extraction, and validation
├── knowledge_base/       # Legal chunking, embedding, and retrieval
├── tests/                # Unit and integration tests
├── ui/                   # Streamlit application and pages
├── docs/                 # Documentation assets (trace screenshots)
├── llm_client.py         # Traced Groq client shared by the agent and ingestion
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
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
- `ingestion/confidence.py` computes extraction confidence from the extracted fields.
- `ingestion/processors.py` orchestrates the ingestion pipeline.

### Agent

- `agent/graph.py` defines the LangGraph workflow, including the deterministic VAT-rate check.
- `agent/tools.py` provides legal retrieval and web search helpers.
- `agent/auditor.py` exposes the public audit entry points.
- `agent/schemas.py` defines the final audit report models.

### Knowledge Base

- `knowledge_base/chunker.py` splits legal text into article-level chunks.
- `knowledge_base/embedder.py` embeds and upserts chunks into Pinecone.
- `knowledge_base/retriever.py` performs hybrid retrieval using vector search and BM25.
- `knowledge_base/loader.py` loads seed data or PDFs into the knowledge base.

### Shared

- `llm_client.py` is the single entry point for Groq calls. Every LLM call is traced in LangSmith when tracing is enabled.

## Requirements

- Python 3.10+
- Pinecone API key for knowledge-base storage and retrieval
- Groq API key for agent reasoning and extraction (model set by `GROQ_MODEL`)
- Tavily API key if web search is enabled
- LangSmith API key (optional) for tracing
- Streamlit for the UI

## Environment Setup

Create a `.env` file from the provided example and fill in your keys:

```bash
copy .env.example .env
```

Variables:

- `GROQ_API_KEY`
- `GROQ_MODEL` (default `openai/gpt-oss-120b`; use a model your Groq account can access)
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (default `financial-auditor`)
- `TAVILY_API_KEY` (optional, enables web search)
- `APP_USERNAME` and `APP_PASSWORD` (UI login; sign-in is not possible without a password)
- `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` (optional, LangSmith tracing)
- `PDF_ARABIC_FONT_PATH` (optional, custom font for PDF export)

Do not put quotes around values: Docker's `--env-file` keeps them as part of the value.

The repository ignores `.env`, so your secrets will not be committed.

## Local Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Load The Knowledge Base

Run this once to create the Pinecone index (1024 dimensions) and load the seed legal articles:

```bash
python -m knowledge_base.loader --mode seed
```

The first run downloads the embedding model, so expect a wait.

## Run The App

Start the Streamlit UI with:

```bash
streamlit run ui/app.py
```

Sign in with the `APP_USERNAME` and `APP_PASSWORD` values from `.env`. The embedding model loads in the background at startup, so the first audit does not pay the load time.

## Run The Tests

Run the full test suite with:

```bash
pytest tests -q
```

Current test behavior:

- Unit tests run locally without external services.
- Integration tests are marked and will skip automatically when Pinecone, Groq, or the embedding stack is unavailable.
- If tracing is enabled in `.env`, test runs also send traces. In PowerShell, run `$env:LANGCHAIN_TRACING_V2="false"` before `pytest`, and `Remove-Item Env:LANGCHAIN_TRACING_V2` afterwards: the setting lasts for the whole terminal session.

## Observability

Every audit is traced end to end with LangSmith: one trace per audit, with a child span for each graph node, each legal search query, and each LLM call (prompt, response, token usage, and `finish_reason`). Document extraction is traced as its own run.

![LangSmith trace of one audit](docs/langsmith-trace.png)

To enable tracing, set these variables in `.env`:

```dotenv
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your key>
LANGCHAIN_PROJECT=financial-auditor
```

Traces include document text, so enable tracing only for non-sensitive documents.

### What the traces found

| Finding | Cause | Fix | Result |
|---|---|---|---|
| First audit took 25.8s | Embedding model loaded on first use (retrieval alone: 22s) | Model is loaded in the background at app start | First audit ≈ 5.4s |
| Invoice with a 12% VAT rate passed | Tax below 14% of the subtotal was treated as exempt items, and the model saw only 300 characters of text | Deterministic check that a stated VAT rate is a legal rate (0 / 5 / 14%) | Both the rate and the missing tax ID are flagged |
| Extraction returned all zeros | The reasoning model spent its token budget on thinking, so the JSON was cut off | Larger token budget, lower reasoning effort for extraction, `finish_reason` logged on every call | Truncation is visible in the trace |
| Every report required manual review | Confidence measured the keyword classifier (55% on a correct invoice) | Confidence computed from extraction evidence: amounts found, subtotal + tax = total, valid tax ID | Correct invoice scores 89% and is auto-approved |

### Future work

- Compute the compliance score in code (model-assigned scores vary between runs).
- Run the legal searches in parallel (about 3s of a 5.4s audit).
- Check VAT per line item instead of per document total.

## Docker

The repository includes a `Dockerfile` for containerized runs of the UI.

Build the image:

```bash
docker build -t ai-financial-auditor .
```

Run the container:

```bash
docker run --rm -p 8501:8501 --env-file .env -e HF_HOME=/models -e SENTENCE_TRANSFORMERS_HOME=/models -v hf_cache:/models ai-financial-auditor
```

The embedding model is downloaded on first start. The `hf_cache` volume keeps it between runs, so later starts do not download it again.

Then open the app at `http://localhost:8501`.

## Notes On Testing And Reliability

- The test suite includes both fast unit tests and guarded integration tests.
- Knowledge-base tests verify chunking, BM25 retrieval, and Pinecone integration behavior.
- Agent tests verify schema validation, tool behavior, and audit flow contracts.
- External-service tests are designed to skip cleanly instead of failing when credentials or runtime dependencies are missing.

## License And Usage

This project is intended for internal or organizational use as a financial-document auditing assistant. Review and adapt the legal and compliance logic before using it in production.
