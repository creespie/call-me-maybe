import json

def ask_prompt_name(prompt: str, json_input, names, call) -> str:
    ret = """<|im_start|>system
You may call one function to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools> \n"""
    for func in json_input:
        if func["name"] not in names:
            continue
        ret += f"{json.dumps(func)}\n"
    ret += f'''
</tools>
Return a json object with the  function name within <tool_call></tool_call> XML tags.
<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<tool_call>
{{"name": "{call}}} 
            '''
    return ret


def ask_prompt_value(prompt, function, parameter, call):
    val = f'''
<|im_start|>system
You may find or complete the {parameter} parameter related to the user query's function
You will find the current parameters in the <tool_call></tool_call> XML tags.
You are provided the function signatures within <tools></tools> XML tags:
<tools>
{json.dumps(function)}
</tools>
Return a json object with the one or more values within <tool_call></tool_call> XML tags.
<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<tool_call>
{call}
        '''
    return val






# PHASE1_TEMPLATE = """<|im_start|>system
# You may call one function to assist with the user query.
# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# <tools>
# {"name": "fn_add_numbers", "description": "Add two numbers together and return their sum.", "parameters": {"a": {"type": "number"}, "b": {"type": "number"}}}
# {"name": "fn_greet", "description": "Generate a greeting message for a person by name.", "parameters": {"name": {"type": "string"}}}
# </tools>
# Return a json object with the  function name within <tool_call></tool_call> XML tags.
# <|im_end|>
# <|im_start|>user
# {USER_PROMPT}<|im_end|>
# <|im_start|>assistant
# <tool_call>
# {{"name": \""""   