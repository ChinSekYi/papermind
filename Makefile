install:
	uv pip install -r requirements.txt

format:
	isort *.py
	black *.py

run:
	python main.py

lint: 
	ruff check src/

all: install format lint