install:
	uv pip install -r requirements.txt

format:
	isort *.py
	black *.py

run:
	uv run python -m src.data.model

api-server:
	fastapi dev backend/main.py

app: 
	streamlit run frontend/app.py
	
lint: 
	ruff check src/

all: install format lint