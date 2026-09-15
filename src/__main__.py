"""Entry point for the function calling project.

Usage:
    uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
"""

import argparse
import json
import sys
import os
import numpy
from pathlib import Path
from typing import Any
from .functions import func_parser, prompt_parser
from .create_prompt import ask_prompt_name, ask_prompt_value
from llm_sdk import Small_LLM_Model


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
    path = Path(path)
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

def normalize_input_ids(input_ids: Any) -> list[int]:
    """Normalize whatever model.encode() returns into a flat list[int]."""
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(i) for i in input_ids]

def check_string(decoded: str) -> bool:

    for char in decoded:
        code = ord(char)

        if code < 0x20:
            return False

    return True

def check_nbr(decoded: str) -> bool:
    int_set = set("0123456789e.\"")
    for char in decoded:
        if char not in int_set:
            return False
    return True


def main() -> None:
    """Run the function calling pipeline end to end."""
    args = parse_args()

    functions_definition = load_json_file(args.functions_definition)
    prompts = load_json_file(args.input)
    model = Small_LLM_Model()

    functions_by_name = {func["name"]: func for func in functions_definition}
    names, params, desc = func_parser(functions_definition)
    prompt_parser(prompts)
    results: list[dict[str, Any]] = []
    for p in prompts:
        call = os.path.commonprefix(names)
        temp_name = names
        #finds function name
        for _ in range(50):
            current_question = model.encode(ask_prompt_name(p["prompt"], functions_definition, temp_name, call))
            current_question = normalize_input_ids(current_question)
            logits = numpy.array(model.get_logits_from_input_ids(current_question))
            sorted_ids = numpy.argsort(logits)[::-1]
            found = False
            for max_id in sorted_ids:
                token_str = model.decode([max_id])
                test_call = call + token_str
                if any(test_call in s for s in temp_name if s.startswith(test_call)):
                    call = test_call
                    found = True
                    break
            print(f"DEBUG name loop: token={token_str!r} call={call!r} found={found} candidates={temp_name}")
            temp_name =[name for name in temp_name if name.startswith(call)]
            if len(temp_name) == 1:
                call = temp_name[0]
                break
        if len(temp_name) != 1:
            results.append({"prompt": p["prompt"], "name": "Unknown", "parameters": "Unknown"})
            continue
        #finds function value
        param_str = "{"
        selected_function = functions_by_name[temp_name[0]]
        param_keys = list(selected_function["parameters"].keys())
        for pa in param_keys:
            is_number = selected_function["parameters"][pa]["type"] in ("number", "int", "float", "digit")
            p_value = ""
            param_str += f'"{pa}": ' if is_number else f'"{pa}": "'
            for _ in range(50):
                current_question = model.encode(ask_prompt_value(p["prompt"], selected_function, pa, param_str + p_value))
                current_question = normalize_input_ids(current_question)
                logits = numpy.array(model.get_logits_from_input_ids(current_question))
                max_id = numpy.argmax(logits)
                new = model.decode([max_id])
                if new == "":
                    break
                if is_number:
                    prefix = ""
                    for ch in new:
                        if ch in "0123456789.eE-":
                            prefix += ch
                        else:
                            break
                    p_value += prefix
                    if len(prefix) < len(new):
                        break
                else:
                    if '"' in new:
                        p_value += new.split('"')[0]
                        break
                    if check_string(new):
                        p_value += new
                    else:
                        break
            param_str += p_value
            if not is_number:
                param_str += '"'
            if param_keys.index(pa) < len(param_keys) - 1:
                param_str += ', '
            else:
                param_str += "}"
        print(f"DEBUG param_str: {param_str!r}")
        results.append({"prompt": p["prompt"], "name": temp_name[0], "parameters": json.loads(param_str)})

    write_json_file(args.output, results)


if __name__ == "__main__":
    main()