# Papermind
Extracts, analyses and compares research papers to generate insights

## Features
- Summary and key insights
- Methodology
- Limitations
- References

## Setup

1. Install Ollama: https://ollama.ai
2. Pull Gemma 4 model: 
   ```bash
   ollama pull gemma4
   ```
   ref: https://ollama.com/library/gemma4
3. Make sure Ollama is running:
   ```bash
   ollama list
   ```
   If this command works, Ollama is already running. If it fails, start it with:
   ```bash
   ollama serve
   ```
4. Install Python dependencies:
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