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
from functions import func_parser, prompt_parser
from create_prompt import ask_prompt_name, ask_prompt_value
from ..llm_sdk.llm_sdk import Small_LLM_Model


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

def check_string(decoded: str) -> bool:

    for char in decoded:
        code = ord(char)

        if code < 0x20:
            return False

    return True

def check_nbr(decoded: str) -> bool:
    int_set = set("0","1","2", "3", "4", "5" , "6" , "7", "8", "9", "e", ".", '"')
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
    vocab = load_json_file(model.get_path_to_vocab_file())

    names, params, desc = func_parser(functions_definition)
    prompt_parser(prompts)
    results: list[dict[str, Any]] = []
    for p in prompts:
        call = os.path.commonprefix(names)
        temp_name = names
        #finds function name
        for _ in range(50):
            current_question = model.encode(ask_prompt_name(p["prompt"], functions_definition, temp_name, call))
            logits = numpy.array(model.get_logits_from_input_ids(current_question))
            for _ in range(50):
                test_call = ""
                max_id = numpy.argmax(logits)
                test_call = call + model.decode([max_id])
                if any(test_call in s for s in temp_name):
                    call = test_call
                    break
                logits[max_id] = -numpy.inf
            temp_name =[name for name in temp_name if name.startswith(call)]
            if len(temp_name) == 1:
                call = temp_name[0]
                break
        if len(temp_name) != 1:
            results.append({"prompt": "Couldn't find it in reasonable time", "name": "Unknown", "parameters": "Unknown"})
            continue
        #finds function value
        param_str = "{"
        selected_function = functions_definition[names.index(temp_name[0])]
        for p in selected_function["parameters"].keys():
            p_value = ""
            param_str += f'"{p}": "{p_value}'
            for _ in range(50):
                current_question = model.encode(ask_prompt_value(p["prompt"], selected_function, p, param_str))
                logits = numpy.array(model.get_logits_from_input_ids(current_question))
                for _ in range(50):
                    max_id = numpy.argmax(logits)
                    new = model.decode([max_id])
                    if selected_function["parameters"][p]["type"] in ("number", "int", "float", "digit"):
                        if check_nbr(new):
                            p_value += new
                            break
                    else:
                        if check_string(new):
                            p_value += new
                            break
                if p_value[:-1] == '"':
                    break
            if selected_function["parameters"].keys().index(p) < len(selected_function["parameters"].keys()) - 1:
                param_str += ', '    
            else:
                param_str += "}"
        results.append({"prompt": p, "name": temp_name[0], "parameters": param_str})

    write_json_file(args.output, results)


if __name__ == "__main__":
    main()