"""Entry point for the function calling project.

Usage:
    uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Root of the project = parent of the "src" package this file lives in.
# __file__ is .../<project_root>/src/__main__.py, so parent.parent is the root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FUNCTIONS_DEFINITION = PROJECT_ROOT / "data" / "input" / "functions_definition.json"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "input" / "function_calling_tests.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "output" / "function_calling_results.json"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with resolved Path objects for
        functions_definition, input, and output.
    """
    parser = argparse.ArgumentParser(
        prog="src",
        description="Translate natural language prompts into structured function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help="Path to the JSON file describing available functions "
        f"(default: {DEFAULT_FUNCTIONS_DEFINITION})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the JSON file with prompts to process (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write the JSON results to (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def load_json_file(path: Path) -> Any:
    """Load and parse a JSON file, raising a clear error on failure.

    Args:
        path: Path to the JSON file to load.

    Returns:
        The parsed JSON content (typically a list or dict).

    Raises:
        SystemExit: If the file is missing or contains invalid JSON.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)


def write_json_file(path: Path, data: Any) -> None:
    """Write data as JSON to the given path, creating parent dirs if needed.

    Args:
        path: Destination path for the JSON file.
        data: JSON-serializable data to write.

    Raises:
        SystemExit: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"Error: could not write output to {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    """Run the function calling pipeline end to end."""
    args = parse_args()

    functions_definition = load_json_file(args.functions_definition)
    prompts = load_json_file(args.input)

    # TODO: validate functions_definition and prompts with pydantic models
    # TODO: for each prompt, run the constrained generation loop against the
    #       LLM to produce {"prompt": ..., "name": ..., "parameters": ...}
    # TODO: collect results into a list

    results: list[dict[str, Any]] = []
    for entry in prompts:
        prompt = entry["prompt"]
        # placeholder until the generation logic is implemented
        result = {"prompt": prompt, "name": None, "parameters": {}}
        results.append(result)

    write_json_file(args.output, results)


if __name__ == "__main__":
    main()