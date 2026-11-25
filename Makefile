PYTHON = uv run python

all: run

run:
	$(PYTHON) main.py app

eval:
	$(PYTHON) main.py eval

experiments:
	$(PYTHON) main.py experiments

install:
	uv sync

test:
	uv run --with pytest pytest -v

lint:
	uv run --with ruff ruff check src tests main.py
