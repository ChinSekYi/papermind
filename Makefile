install:
	uv pip install -r requirements.txt

format:
	isort *.py
	black *.py

lint: 
	ruff check src/

run:
	uv run python -m src.data.model

demo:
	uv run python -m src.data.demo

api-server:
	fastapi dev backend/main.py

app: 
	streamlit run frontend/app.py
	
lint: 
	ruff check src/

all: install format lint