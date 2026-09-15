from llm_sdk import Small_LLM_Model
import json
import numpy

test = Small_LLM_Model()
print(test.encode("<|im_start|>"))
logits = numpy.array(test.get_logits_from_input_ids(test.encode("3 + 2 =").tolist()[0]))
max_id = int(numpy.argmax(logits))
print(max_id)
with open(test.get_path_to_tokenizer_file(), "r", encoding="utf-8") as f:    
    file = json.load(f)
print(file.keys())
print(file["pre_tokenizer"])
with open("data/input/functions_definition.json", "r", encoding="utf-8") as f:
    func = json.load(f)
print(func[0]["parameters"].keys())
# print(list(file.values())[max_id])
# print(test.decode([1]))
# print(test.encode('"'))
# # 151642
# print(len(logits))