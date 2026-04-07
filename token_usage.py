from transformers import AutoTokenizer

def count_tokens(text, model_name="google-t5/t5-base"):
    if not text:
        return 0

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))