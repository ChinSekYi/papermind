# Papermind

## Objective
Papermind is a RAG pipeline optimized for dense academic papers, that helps junior engineers understand prerequisites of a paper they want to read — by extracting clean, complete context that generic tools miss.

Expected Features:
- Section-by-section plain-language explanations
- Prerequisite concept detection
- “Read next” learning links
- Paper understanding summary
- Confidence flags for uncertain explanations

---

### Phase 1: Document parsing benchmark for RAG pipelines.
Document parsers vary significantly in quality—especially when extracting tables, equations, and images. This phase compares parsing strategies to identify the best parser for production RAG systems.

Benchmarking 3 parsing strategies on complex research papers:
- **pymupdf4llm** — Fast, basic text extraction
- **Docling** — Rich structure preservation (tables, images)
- **MinerU** — High-quality OCR/VLM-based extraction

Evaluation metrics: table/equation/image extraction accuracy, structure preservation, latency.
Tracked in MLflow with manual quality scoring and artifact comparison.


## Tech stack

- Python 3.11+
- PyTorch (used by some models)
- `pymupdf4llm`, `docling`, `mineru` (parsers)
- MLflow 
- FastAPI (backend) and Streamlit (frontend)
- `uv` / repo venv helper and GNU `make`
- Optional: Ollama for local LLM hosting


## Setup (minimal)

- (Optional) If you use Ollama: https://ollama.ai
- Install Python dependencies:
```bash
uv sync
source .venv/bin/activate
```

## How to Run
```bash
# terminal 1
make api-server      # starts FastAPI backend on http://127.0.0.1:8000

# terminal 2
make app             # starts Streamlit frontend on http://127.0.0.1:8501
```

Quick inspect (open MLflow UI):
```bash
mlflow ui --backend-store-uri notebooks/mlruns
```