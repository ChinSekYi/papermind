# Detailed Setup

This document contains the full setup and run steps so you don't lose the detailed instructions removed from the top-level README.

## 1. (Optional) Ollama
If you use Ollama for local LLM hosting:

- Install: https://ollama.ai
- Pull the Gemma 4 model (optional):
```bash
ollama pull gemma4
```
- Verify Ollama is running:
```bash
ollama list
```
- Start Ollama if needed:
```bash
ollama serve
```

## 2. Python environment
The project uses a lightweight virtual environment workflow.

- Sync/create the venv and install dependencies:
```bash
uv sync
```
(This runs the repo's environment setup; if `uv` is not available, create a venv manually and install from requirements.)

- Activate the environment:
```bash
source .venv/bin/activate
```

- If you need to install manually:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. MLflow (local file store)
Runs are tracked to the file store at `notebooks/mlruns` by default.

- Launch MLflow UI to inspect runs and artifacts:
```bash
mlflow ui --backend-store-uri notebooks/mlruns
```

- The benchmark notebook writes artifacts to `experiments/outputs/` and logs runs to `notebooks/mlruns`.

## 4. Running the project
- Start the API server (FastAPI):
```bash
make api-server
# starts on http://127.0.0.1:8000
```

- Start the Streamlit frontend:
```bash
make app
# opens on http://127.0.0.1:8501
```

- Run the parsing benchmark notebook (example):
  - Open `notebooks/parsing_benchmark.ipynb` in Jupyter/VS Code and run the cells.
  - Or run the helper wrapper in Python if available in the repo.

## 5. Reproducing a benchmark run
1. Put target PDFs under `data/raw/`.
2. Open `notebooks/parsing_benchmark.ipynb` and run the notebook cells (cells 2–7 are the quick start for the benchmark).
3. After runs complete, open the MLflow UI to inspect artifacts and logs:
```bash
mlflow ui --backend-store-uri notebooks/mlruns
```

## 6. Notes on parsers and known issues
- Available parsers in Phase 1: `pymupdf4llm`, `docling` (basic text extraction), `mineru`.
- Docling formula-enrichment was attempted but caused compatibility issues in some environments; Phase 1 uses Docling for text extraction only. If you want formula enrichment, see model compatibility notes in the repo issues or re-enable in the notebook after resolving HF transformer/processor versions.

## 7. Troubleshooting
- If `uv sync` fails, follow the manual venv steps above.
- If `mlflow` is not installed, install it into the venv:
```bash
pip install mlflow
```
- If a parser executable (e.g. `mineru`) is required but not installed, follow that tool's installation instructions and make sure it is on your PATH.

---

Keep this file as the authoritative place for detailed setup steps; the top-level `README.md` is intentionally concise for quick scanning by hiring managers.
