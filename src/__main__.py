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
from .create_prompt import ask_prompt
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

def common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the shared leading token-id sequence between a and b."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def log_softmax(logits: numpy.ndarray) -> numpy.ndarray:
    m = logits.max()
    return logits - m - numpy.log(numpy.sum(numpy.exp(logits - m)))


def score_candidate(model, prompt: str, base_ids: list[int]) -> float:
    """Teacher-forced sum of log P(token | context) for the tokens in `prompt`
    that extend beyond the shared `base_ids` prefix."""
    cand_ids = normalize_input_ids(model.encode(prompt))
    start = common_prefix_len(base_ids, cand_ids)
    score = 0.0
    for i in range(start, len(cand_ids)):
        context = cand_ids[:i]
        logits = numpy.array(model.get_logits_from_input_ids(context))
        score += float(log_softmax(logits)[cand_ids[i]])
    return score


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
        base_prefix = os.path.commonprefix(names)
        call = base_prefix
        temp_name = names

        #finds function name: cheap single-token race first (1 forward pass)
        generated = f'{{"name": "{call}'
        current_question = model.encode(ask_prompt(p["prompt"], functions_definition, temp_name, generated))
        current_question = normalize_input_ids(current_question)
        logits = numpy.array(model.get_logits_from_input_ids(current_question))
        sorted_ids = numpy.argsort(logits)[::-1]
        for max_id in sorted_ids:
            token_str = model.decode([max_id])
            test_call = call + token_str
            if any(s.startswith(test_call) for s in temp_name):
                call = test_call
                break
        picked = [n for n in temp_name if n.startswith(call)][0]

        # A single BPE token can merge past the point where two names diverge
        # (e.g. 'g' -> greet vs get_square_root), landing confidently on the
        # wrong one. Cheap check (no model calls): does `picked` belong to a
        # group of names sharing the same next character after the common
        # prefix? Only if so, verify with full teacher-forced scoring,
        # restricted to that small group.
        next_char = picked[len(base_prefix):len(base_prefix) + 1]
        collision_group = [n for n in names if n[len(base_prefix):len(base_prefix) + 1] == next_char]

        if len(collision_group) > 1:
            base_generated = f'{{"name": "{base_prefix}'
            base_prompt = ask_prompt(p["prompt"], functions_definition, collision_group, base_generated)
            base_ids = normalize_input_ids(model.encode(base_prompt))
            scores = {}
            for candidate in collision_group:
                cand_generated = f'{{"name": "{candidate}'
                cand_prompt = ask_prompt(p["prompt"], functions_definition, collision_group, cand_generated)
                scores[candidate] = score_candidate(model, cand_prompt, base_ids)
            print(f"DEBUG collision group {collision_group}, scores: {scores}")
            picked = max(scores, key=scores.get)

        call = picked
        temp_name = [call]

        #finds function arguments, continuing the SAME generated <tool_call> sequence
        selected_function = functions_by_name[temp_name[0]]
        param_keys = list(selected_function["parameters"].keys())
        arguments_str = "{"
        for idx, pa in enumerate(param_keys):
            is_number = selected_function["parameters"][pa]["type"] in ("number", "int", "float", "digit")
            p_value = ""
            arguments_str += f'"{pa}": ' if is_number else f'"{pa}": "'
            for _ in range(50):
                generated = f'{{"name": "{temp_name[0]}", "arguments": {arguments_str}{p_value}'
                current_question = model.encode(ask_prompt(p["prompt"], functions_definition, [temp_name[0]], generated))
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
            arguments_str += p_value.strip()
            if not is_number:
                arguments_str += '"'
            arguments_str += ', ' if idx < len(param_keys) - 1 else '}'
        print(f"DEBUG arguments: {arguments_str!r}")
        results.append({"prompt": p["prompt"], "name": temp_name[0], "parameters": json.loads(arguments_str)})

    write_json_file(args.output, results)


if __name__ == "__main__":
    main()