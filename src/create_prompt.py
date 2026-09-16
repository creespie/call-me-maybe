import json
from typing import Any, Collection


def ask_prompt(prompt: str, json_input: list[dict[str, Any]],
               names: Collection[str],
               generated: str,) -> str:
    """Build the full Hermes-style prompt, ending with `generated` (the
    <tool_call> JSON produced so far: name and, once known, arguments)."""
    ret = """<|im_start|>system
You may call one function to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools> \n"""
    for func in json_input:
        if func["name"] not in names:
            continue
        ret += f"{json.dumps(func)}\n"
    ret += """
</tools>
For each function call, return a json object with function name and arguments w
ithin <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

For example, if a "fn_add_numbers" function existed, taking parameters "a" and
 "b":
<tool_call>
{"name": "fn_add_numbers", "arguments": {"a": 1, "b": 2}}
</tool_call>
<|im_end|>
"""
    ret += f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<tool_call>
{generated}"""
    return ret
