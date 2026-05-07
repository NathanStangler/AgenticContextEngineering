import torch
torch.backends.mps.is_available = lambda: False

from summarization import Summarization

s = Summarization()

while True:
    try:
        text = input()
        if not text:
            break
        print(s.summarize([text], 120, 40), flush=True)
    except EOFError:
        break