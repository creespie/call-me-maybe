import json
from llm_sdk import Small_LLM_Model

def main():
    model = Small_LLM_Model()
    enc = model.encode("ciao")

    print(enc)
    print(model.decode(enc))
    print(model.decode([66]))
    print(max(model.get_logits_from_input_ids([66, 22516])))
    path = model.get_path_to_tokenizer_file()
    with open(path, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)
    print(tokenizer.keys())
    print(tokenizer['added_tokens'])

if __name__ == "__main__":
    main()