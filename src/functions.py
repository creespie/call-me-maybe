import sys

def func_parser(json_input: dict[str, str | dict[str, str]]) -> list[str]:
    names = []
    params = []
    desc = []
    for func in json_input:
        for k in ("name", "description", "parameters"):
            if k not in func.keys():
                sys.exit(f"Missing key in {func}")
        if not func["name"] or not isinstance(func["name"], str):
            sys.exit(f"Missing value in {func['name']}")
        elif not func["description"] or not isinstance(func["description"], str):
            sys.exit(f"Missing value in {func['description']}")
        elif not func["parameters"] or not isinstance(func["parameters"], dict):
            sys.exit(f"Missing or wrong value in {func['parameters']}")
        for param in func["parameters"].keys():
            if "type" not in func["parameters"][param]:
                sys.exit(f"Missing type in {param}")
            elif not func["parameters"][param]["type"] or not isinstance(func["parameters"][param]["type"], str):
                sys.exit(f"Missing value in {func['parameters'][param]['type']}")
        names.append(func["name"])
        params.append(func["parameters"])
        desc.append(func["description"])
    return names, params, desc

def prompt_parser(json_input: dict[str, str]):
    for func in json_input:
        if "prompt" not in func.keys():
            sys.exit(f"Missing prompt")
        elif not func["prompt"] or not isinstance(func["prompt"], str):
            sys.exit(f"Missing value in {func["prompt"]}")
