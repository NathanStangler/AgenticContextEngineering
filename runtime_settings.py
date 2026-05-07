import os
# Was dealing with semaphore errors when launching
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

try:
    from tqdm import tqdm

    tqdm.monitor_interval = 0
except Exception:
    pass
