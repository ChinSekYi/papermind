install:
	uv pip install -r requirements.txt

format:
	isort *.py
	black *.py

run:
	uv run python -m src.data.model

lint: 
	ruff check src/

all: install format lint