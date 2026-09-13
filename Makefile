PYTHON  := python3
UV = uv

.PHONY: install run debug clean lint lint-strict

install:
	$(UV) sync

run:
	uv run python -m src [--functions_definition ...] [--input ...] [--output ...]

debug:
	$(PYTHON) -m pdb main.py $(MAP)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	flake8 .
	mypy . 

lint-strict:
	flake8 .
	mypy . --strict