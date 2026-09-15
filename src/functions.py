import sys

def func_parser(json_input: dict[str, str | dict[str, str]]) -> list[str]:
    names = []
    params = []
    desc = []
    for func in json_input:
        if ("name", "description", "parameters") not in func.keys():
            sys.exit(f"Missing key in {func}")
        elif not func["name"] or not isinstance(func["name"], str):
            sys.exit(f"Missing value in {func["name"]}")
        elif not func["description"] or not isinstance(func["description"], str):
            sys.exit(f"Missing value in {func["description"]}")
        elif not func["parameters"] or not isinstance(func["parameters"], dict):
            sys.exit(f"Missing or wrong value in {func["parameters"]}")
        for param in func["parameters"]:
            if "type" not in param.keys():
                sys.exit(f"Missing type in {param}")
            elif not param["type"] or not isinstance(param["type"], str):
                sys.exit(f"Missing value in {param["type"]}")
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
