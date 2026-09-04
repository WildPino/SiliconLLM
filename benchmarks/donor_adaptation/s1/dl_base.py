import sys, json
from huggingface_hub import snapshot_download
for rid in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B"):
    try:
        p = snapshot_download(rid, allow_patterns=["*.json","*.safetensors","*.txt","*.model"])
        print("OK", rid, p, flush=True)
    except Exception as e:
        print("FAIL", rid, type(e).__name__, e, flush=True)
