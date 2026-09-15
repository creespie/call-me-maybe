PYTHON := python3
UV     := uv

# Override from the command line, e.g.:
#   make run INPUT=data/input/my_prompts.json
FUNCTIONS_DEFINITION ?=
INPUT                ?=
OUTPUT               ?=

RUN_ARGS := $(if $(FUNCTIONS_DEFINITION),--functions_definition $(FUNCTIONS_DEFINITION)) \
            $(if $(INPUT),--input $(INPUT)) \
            $(if $(OUTPUT),--output $(OUTPUT))

.PHONY: install run debug clean lint lint-strict

install:
	$(UV) sync

run:
	$(UV) run $(PYTHON) -m src $(RUN_ARGS)

debug:
	$(UV) run $(PYTHON) -m pdb -m src $(RUN_ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	flake8 .
	mypy .

lint-strict:
	flake8 .
	mypy . --strict