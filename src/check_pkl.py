#!/usr/bin/env python3
import os
import sys
import json
from collections import Counter, defaultdict

import torch

CKPT_PATH = "/home/johnny305/Documents/dfpaint/difix_sls/bs1_2gpu/model_53001.pkl"
MAX_SHOW_KEYS = 50  # limit printed keys

def unwrap_state_dict(obj):
    """
    Many trainers save {'state_dict': {...}} or {'model': {...}}.
    Return the inner state_dict if present; otherwise return obj unchanged.
    """
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    return obj

def is_probably_lora(keys):
    probe = [k for k in keys if "lora" in k.lower() or k.endswith(("lora_up.weight","lora_down.weight"))]
    return len(probe) > 0

def prefix_counts(keys, depth=1):
    """
    Count key prefixes up to `depth` dot-levels.
    depth=1 => 'unet', 'vae', 'text_encoder', ...
    depth=2 => 'unet.conv1', 'unet.down_blocks', ...
    """
    counts = Counter()
    for k in keys:
        parts = k.split(".")
        if len(parts) >= depth:
            pref = ".".join(parts[:depth])
        else:
            pref = k
        counts[pref] += 1
    return counts

def shape_summary(sd, max_items=50):
    """
    Produce a small list of (key, shape, dtype) entries for inspection.
    """
    out = []
    for i, (k, v) in enumerate(sd.items()):
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            out.append({"key": k, "shape": list(v.shape), "dtype": str(v.dtype)})
        else:
            out.append({"key": k, "type": str(type(v))})
        if i + 1 >= max_items:
            break
    return out

def main():
    path = CKPT_PATH if len(sys.argv) == 1 else sys.argv[1]
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        sys.exit(1)

    print(f"Loading checkpoint (CPU) from: {path}")
    ckpt = torch.load(path, map_location="cpu")
    print(f"Top-level object type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())[:20]}")
    else:
        print("Top-level object is not a dict; might be a pickled class. Will proceed anyway.")

    sd = unwrap_state_dict(ckpt)
    print(f"After unwrap -> type: {type(sd)}")

    if not isinstance(sd, dict):
        print("\n[Result] Not a dict-like state_dict. It may be a pickled model/trainer object.")
        print("Action: You’ll likely need the original code to reconstruct the model and load this file,")
        print("        or export/convert it into a pure state_dict/diffusers format.")
        return

    # Keys overview
    keys = list(sd.keys())
    n_keys = len(keys)
    print(f"\n# of parameters in state_dict: {n_keys}")

    # Quick LoRA check
    if is_probably_lora(keys):
        print("[Hint] Detected LoRA-like keys (contain 'lora'/'lora_up'/'lora_down').")
        print("       Likely loadable via pipe.load_lora_weights(...) or as a LoRA adapter.")
    
    # Prefix counts (depth 1 and depth 2)
    p1 = prefix_counts(keys, depth=1).most_common()
    p2 = prefix_counts(keys, depth=2).most_common()

    print("\nPrefix counts (depth=1) — top 20:")
    for k, c in p1[:20]:
        print(f"  {k:<30} {c}")

    print("\nPrefix counts (depth=2) — top 20:")
    for k, c in p2[:20]:
        print(f"  {k:<30} {c}")

    # Show a sample of raw keys
    print(f"\nSample {min(MAX_SHOW_KEYS, n_keys)} keys:")
    for k in keys[:MAX_SHOW_KEYS]:
        print(" ", k)

    # Shapes/dtypes sample
    sample = shape_summary(sd, max_items=MAX_SHOW_KEYS)
    print("\nSample param shapes/dtypes:")
    for item in sample:
        if "shape" in item:
            print(f"  {item['key']}: shape={item['shape']}, dtype={item['dtype']}")
        else:
            print(f"  {item['key']}: type={item['type']}")

    # Save a compact JSON report next to the checkpoint
    report = {
        "path": path,
        "num_keys": n_keys,
        "is_lora_like": is_probably_lora(keys),
        "top_prefixes_depth1": p1[:50],
        "top_prefixes_depth2": p2[:50],
        "sample": sample,
    }
    rep_path = os.path.splitext(path)[0] + "_inspect.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote summary JSON → {rep_path}")

if __name__ == "__main__":
    main()

