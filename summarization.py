import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarization:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def _trim_to_tokens(self, text, max_tokens):
        if max_tokens <= 0:
            return ""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        trimmed = token_ids[:max_tokens]
        return self.tokenizer.decode(trimmed, skip_special_tokens=True).strip()

    def generate(self, input_text, max_new_tokens, min_new_tokens):
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        min_new_tokens = max(1, min(min_new_tokens, max_new_tokens))

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return self._trim_to_tokens(decoded, max_new_tokens)

    def summarize(self, chunks, max_tokens=128, min_tokens=30):
        chunks = [c.strip() for c in chunks if c and c.strip()]
        if not chunks:
            return ""

        if len(chunks) == 1:
            return self.generate(chunks[0], max_tokens, min_tokens)

        per_chunk_max = max(32, min(80, max_tokens // max(1, len(chunks))))
        per_chunk_min = max(8, min(20, min_tokens // max(1, len(chunks))))

        summaries = []
        for chunk in chunks:
            s = self.generate(chunk, per_chunk_max, per_chunk_min)
            if s:
                summaries.append(s)

        if not summaries:
            return ""

        combined = "\n".join(summaries)
        return self.generate(combined, max_tokens, min_tokens)