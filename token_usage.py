import os
from functools import lru_cache
from transformers import AutoTokenizer

@lru_cache(maxsize=4)
def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def count_tokens(text, model_name="google-t5/t5-base"):
    if not text:
        return 0

    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))