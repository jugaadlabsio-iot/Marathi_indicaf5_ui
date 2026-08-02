# -*- coding: utf-8 -*-
"""Blend a fine-tuned checkpoint back toward the model it started from.

WHY
---
IndicF5 already knew Marathi. It was trained on hundreds of hours and could
say थांबलं and लायब्ररी correctly. The fine-tune was only meant to add one
speaker's timbre - but 80 epochs over 1.26 hours of audio is enough to
overwrite what the base knew. That is catastrophic forgetting, and it explains
the pattern exactly: the voice is right, the Marathi is not.

The cheap fix is weight interpolation, sometimes called a model soup:

    merged = alpha * finetuned + (1 - alpha) * base

Voice identity turns out to live in a fairly small, low-rank part of the
weights, while general pronunciation is spread across all of them. So partial
blends often keep most of the speaker while recovering most of the base
model's competence. It costs one pass of tensor arithmetic - no GPU, no
training, no new recordings.

    python tools/merge_ckpt.py --base base.pt --tuned model_last_slim.pt --alpha 0.6
    python tools/merge_ckpt.py --base base.pt --tuned model_last_slim.pt --sweep

`--sweep` writes several blends so you can listen and pick where the voice
starts to go and the pronunciation comes back. Start around 0.5-0.7.
"""
import argparse
import sys
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--base", required=True, help="the checkpoint the fine-tune started from")
p.add_argument("--tuned", required=True, help="your fine-tuned checkpoint")
p.add_argument("--alpha", type=float, default=0.6,
               help="1.0 = pure fine-tune, 0.0 = pure base")
p.add_argument("--sweep", action="store_true",
               help="write blends at 0.3 0.5 0.7 0.85 instead of one alpha")
p.add_argument("--out-dir", default=None)
a = p.parse_args()

out_dir = Path(a.out_dir) if a.out_dir else Path(a.tuned).parent
KEYS = ("ema_model_state_dict", "model_state_dict")


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not any(k in d for k in KEYS):
        # a raw state dict, wrap it so both shapes are handled the same way
        d = {"ema_model_state_dict": d}
    return d


print(f"base  : {a.base}")
print(f"tuned : {a.tuned}")
base, tuned = load(a.base), load(a.tuned)

alphas = [0.3, 0.5, 0.7, 0.85] if a.sweep else [a.alpha]
for alpha in alphas:
    merged, stats = {}, {"blended": 0, "tuned_only": 0, "shape_mismatch": 0}
    for key in KEYS:
        if key not in tuned:
            continue
        bsd = base.get(key) or base.get(KEYS[0]) or {}
        out = {}
        for name, t in tuned[key].items():
            b = bsd.get(name)
            if b is None or not torch.is_tensor(t):
                out[name] = t
                stats["tuned_only"] += 1
            elif b.shape != t.shape:
                out[name] = t                       # vocab-size layers differ
                stats["shape_mismatch"] += 1
            elif not t.is_floating_point():
                out[name] = t                       # step counters, flags
                stats["tuned_only"] += 1
            else:
                out[name] = (alpha * t.float() + (1 - alpha) * b.float()).to(t.dtype)
                stats["blended"] += 1
        merged[key] = out

    name = Path(a.tuned).stem + f"_merged{int(alpha*100):02d}.pt"
    path = out_dir / name
    torch.save(merged, path)
    size = path.stat().st_size / 1e9
    print(f"  alpha {alpha:4.2f} -> {name}  ({size:.2f} GB)  "
          f"blended {stats['blended']}, kept-from-tuned "
          f"{stats['tuned_only'] + stats['shape_mismatch']}")

print("\nRender the same sentence from each and listen for two things:")
print("  does it still sound like you, and does it say the hard words right.")
print("The useful alpha is the lowest one that still sounds like you.")
