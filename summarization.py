import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarization:
    def __init__(self, model_name="google-t5/t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def summarize(self, chunks, max_tokens, min_tokens):
        chunks = [c.strip() for c in chunks if c and c.strip()]
        if not chunks:
            return ""

        context = '\n'.join(f'- {chunk}' for chunk in chunks)
        prompt = (
            "Summarize the following context for future agentic responses. "
            "Keep facts, user preferences, constraints, and uncompleted tasks. "
            "Remove repetition and keep it concise.\n\n"
            f"Context:\n{context}\n\nSummary:"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            do_sample=False,
            early_stopping=True
        )
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return decoded.strip()